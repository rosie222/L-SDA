import os
import textwrap
import time
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from rich import print
try:
    from openai import APIConnectionError, APITimeoutError
except ImportError:
    # Compatibility with older versions of the openai library
    try:
        from openai.error import APIConnectionError, APITimeoutError
    except ImportError:
        # Define fallback exception classes
        class APIConnectionError(Exception):
            pass
        class APITimeoutError(Exception):
            pass

from lsda.scenario.envScenario import EnvScenario


# Retry configuration constants
RETRY_CONFIG = {
    'max_retries': 5,           # Maximum retry attempts
    'initial_delay': 1.0,       # Initial retry delay (seconds)
    'max_delay': 30.0,          # Maximum retry delay (seconds)
    'backoff_factor': 2.0,      # Exponential backoff factor
    'timeout': 30.0,            # API call timeout (seconds)
}


class DrivingMemory:

    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type
        if encode_type == 'sce_encode':
            # 'sce_encode' is deprecated for now.
            raise ValueError("encode_type sce_encode is deprecated for now.")
        elif encode_type == 'sce_language':
            # Use OpenAI Embeddings with timeout configuration
            self.embedding = OpenAIEmbeddings(
                request_timeout=RETRY_CONFIG['timeout']
            )
            db_path = os.path.join(
                './db', 'chroma_5_shot_20_mem/') if db_path is None else db_path
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path
            )
        else:
            raise ValueError(
                "Unknown ENCODE_TYPE: should be sce_encode or sce_language")
        print(f"==========Loaded {db_path} Memory, Database has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.==========")

    def _exponential_backoff_retry(self, func, *args, max_retries=None, **kwargs):
        """
        Execute a function with exponential backoff retry mechanism.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            max_retries: Maximum number of retry attempts
            **kwargs: Keyword arguments
            
        Returns:
            Function execution result
        """
        if max_retries is None:
            max_retries = RETRY_CONFIG['max_retries']
        
        delay = RETRY_CONFIG['initial_delay']
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                print(f"[cyan]Executing operation (attempt {attempt + 1}/{max_retries})...[/cyan]")
                return func(*args, **kwargs)
            except (APIConnectionError, APITimeoutError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"[yellow]⚠️ Connection error: {type(e).__name__} - {str(e)[:100]}[/yellow]")
                    print(f"[yellow]   Waiting {delay:.1f} seconds before retry... (attempt {attempt + 1}/{max_retries})[/yellow]")
                    time.sleep(delay)
                    # Exponential backoff: delay = min(delay * backoff_factor, max_delay)
                    delay = min(delay * RETRY_CONFIG['backoff_factor'], RETRY_CONFIG['max_delay'])
                else:
                    print(f"[red]❌ Operation failed - exhausted all retry attempts ({max_retries})[/red]")
            except Exception as e:
                print(f"[red]❌ Unexpected error: {type(e).__name__} - {str(e)[:100]}[/red]")
                raise
        
        # All retries failed
        raise last_exception or Exception("Unknown error: all retries exhausted")

    def retriveMemory(self, driving_scenario: EnvScenario, frame_id: int, top_k: int = 5):
        """
        Retrieve similar scenarios from memory with automatic retry support.
        
        Args:
            driving_scenario: Driving scenario object
            frame_id: Frame ID
            top_k: Number of top similar results to return
            
        Returns:
            List of similar results; empty list if retrieval fails
        """
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            try:
                query_scenario = driving_scenario.describe(frame_id)
                
                # Execute similarity search with retry mechanism
                def search_with_score():
                    return self.scenario_memory.similarity_search_with_score(
                        query_scenario, k=top_k
                    )
                
                similarity_results = self._exponential_backoff_retry(search_with_score)
                
                fewshot_results = []
                for idx in range(0, len(similarity_results)):
                    # print(f"similarity score: {similarity_results[idx][1]}")
                    fewshot_results.append(similarity_results[idx][0].metadata)
                
                print(f"[green]✅ Successfully retrieved {len(fewshot_results)} similar scenarios[/green]")
                return fewshot_results
                
            except Exception as e:
                print(f"[red]❌ Memory retrieval failed: {type(e).__name__} - {str(e)[:150]}[/red]")
                print(f"[yellow]⚠️ Returning empty list and continuing execution...[/yellow]")
                return []  # Return empty list instead of interrupting

    def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None, comments: str = ""):
        """
        Add or update memory record with automatic retry support.
        
        Args:
            sce_descrip: Scenario description
            human_question: Question
            response: Response
            action: Action
            sce: Scenario object
            comments: Comments
        """
        try:
            if self.encode_type == 'sce_encode':
                pass
            elif self.encode_type == 'sce_language':
                sce_descrip = sce_descrip.replace("'", '')
            
            # https://docs.trychroma.com/usage-guide#using-where-filters
            get_results = self.scenario_memory._collection.get(
                where_document={
                    "$contains": sce_descrip
                }
            )
            # print("get_results: ", get_results)

            if len(get_results['ids']) > 0:
                # Already exists
                id = get_results['ids'][0]
                self.scenario_memory._collection.update(
                    ids=id, metadatas={"human_question": human_question,
                                       'LLM_response': response, 'action': action, 'comments': comments}
                )
                print(f"[green]✅ Memory item updated. Database now has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.[/green]")
            else:
                doc = Document(
                    page_content=sce_descrip,
                    metadata={"human_question": human_question,
                              'LLM_response': response, 'action': action, 'comments': comments}
                )
                # Add document with retry mechanism
                def add_doc():
                    return self.scenario_memory.add_documents([doc])
                
                id = self._exponential_backoff_retry(add_doc)
                print(f"[green]✅ Memory item added. Database now has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.[/green]")
        
        except (APIConnectionError, APITimeoutError, TimeoutError) as e:
            print(f"[yellow]⚠️ Memory addition failed (network error): {type(e).__name__}[/yellow]")
            print(f"[yellow]   Will retry on next successful operation[/yellow]")
        except Exception as e:
            print(f"[red]❌ Memory addition failed: {type(e).__name__} - {str(e)[:100]}[/red]")

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)
        print(f"Deleted {len(ids)} memory items. Database now has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")

    def editMemory(self, record_id: str, human_question: str = None, response: str = None, action: int = None, comments: str = None):
        """Edit metadata of a specific memory record"""
        try:
            # Get current record
            result = self.scenario_memory._collection.get(ids=[record_id], include=['metadatas'])
            if len(result['ids']) == 0:
                print(f"[yellow]Warning: Record {record_id} not found[/yellow]")
                return
            
            current_metadata = result['metadatas'][0]
            # Update only provided fields
            updated_metadata = current_metadata.copy()
            if human_question is not None:
                updated_metadata['human_question'] = human_question
            if response is not None:
                updated_metadata['LLM_response'] = response
            if action is not None:
                updated_metadata['action'] = action
            if comments is not None:
                updated_metadata['comments'] = comments
            
            # Update the record
            self.scenario_memory._collection.update(ids=[record_id], metadatas=[updated_metadata])
            print(f"[green]Successfully edited record {record_id}[/green]")
        except Exception as e:
            print(f"[red]Error editing memory: {e}[/red]")

    def getMemory(self, record_id: str = None, limit: int = None):
        """Get memory records by ID or get all records with optional limit"""
        try:
            if record_id:
                result = self.scenario_memory._collection.get(ids=[record_id], include=['documents', 'metadatas'])
                if len(result['ids']) == 0:
                    print(f"[yellow]Record {record_id} not found[/yellow]")
                    return None
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            else:
                # Get all records
                limit = limit or 1000
                result = self.scenario_memory._collection.get(
                    include=['documents', 'metadatas'],
                    limit=limit
                )
                records = []
                for i in range(len(result['ids'])):
                    records.append({
                        'id': result['ids'][i],
                        'content': result['documents'][i],
                        'metadata': result['metadatas'][i]
                    })
                return records
        except Exception as e:
            print(f"[red]Error retrieving memory: {e}[/red]")
            return None

    def deleteByComment(self, comment: str):
        """Delete all records with a specific comment."""
        try:
            result = self.scenario_memory._collection.get(
                where={"comments": {"$eq": comment}},
                include=['metadatas']
            )
            if len(result['ids']) > 0:
                self.scenario_memory._collection.delete(ids=result['ids'])
                print(f"[green]✅ Deleted {len(result['ids'])} records with comment '{comment}'. Database now has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.[/green]")
            else:
                print(f"[yellow]No records found with comment '{comment}'[/yellow]")
        except Exception as e:
            print(f"[red]Error deleting records: {e}[/red]")

    def getStatistics(self):
        """Get database statistics."""
        try:
            all_records = self.scenario_memory._collection.get(include=['metadatas'])
            total = len(all_records['ids'])
            
            # Count by comments
            comment_counts = {}
            action_counts = {}
            for metadata in all_records['metadatas']:
                comment = metadata.get('comments', 'unknown')
                action = metadata.get('action', 'unknown')
                comment_counts[comment] = comment_counts.get(comment, 0) + 1
                action_counts[action] = action_counts.get(action, 0) + 1
            
            stats = {
                'total_records': total,
                'by_comment': comment_counts,
                'by_action': action_counts
            }
            print(f"[cyan]Database Statistics: {stats}[/cyan]")
            return stats
        except Exception as e:
            print(f"[red]Error getting statistics: {e}[/red]")
            return None

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        for i in range(0, len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] not in current_documents['embeddings']:
                self.scenario_memory._collection.add(
                    embeddings=other_documents['embeddings'][i],
                    metadatas=other_documents['metadatas'][i],
                    documents=other_documents['documents'][i],
                    ids=other_documents['ids'][i]
                )
        print(f"Merge complete. Database now has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")
    
    def persist(self):
        """Explicitly persist the database to disk."""
        try:
            if hasattr(self.scenario_memory, 'persist'):
                self.scenario_memory.persist()
            print(f"[green]✅ Database persisted successfully[/green]")
        except Exception as e:
            print(f"[yellow]Warning: Failed to persist database: {e}[/yellow]")


if __name__ == "__main__":
    pass