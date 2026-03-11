#!/usr/bin/env python3
"""
Database Compaction Tool
Re-index and compact ChromaDB to create continuous IDs
This is optional - only use if you need continuous ID numbering
"""

import argparse
import json
from lsda.driver_agent.vectorStore import DrivingMemory
from langchain.docstore.document import Document


def compact_database(db_path, output_path=None):
    """
    Compact database by re-indexing with continuous IDs
    
    Creates a new database with continuous IDs while preserving all data
    Original database is NOT modified
    """
    
    memory = DrivingMemory(db_path=db_path)
    
    # Get all records
    all_records = memory.scenario_memory._collection.get(
        include=['documents', 'metadatas']
    )
    
    total_records = len(all_records['ids'])
    print(f"[cyan]Found {total_records} records to compact[/cyan]")
    
    if total_records == 0:
        print("[yellow]Database is empty, nothing to compact[/yellow]")
        return
    
    # Create new memory with compacted IDs
    output_path = output_path or db_path.rstrip('/') + '_compacted/'
    print(f"[cyan]Creating compacted database at: {output_path}[/cyan]")
    
    new_memory = DrivingMemory(db_path=output_path)
    
    # Re-add all records with new continuous IDs
    for idx, (old_id, doc, metadata) in enumerate(zip(
        all_records['ids'],
        all_records['documents'],
        all_records['metadatas']
    )):
        new_id = f"record_{idx + 1:06d}"  # New continuous ID: record_000001, record_000002, etc.
        
        doc_obj = Document(page_content=doc, metadata=metadata)
        new_memory.scenario_memory.add_documents(
            [doc_obj],
            ids=[new_id]
        )
        
        if (idx + 1) % 10 == 0:
            print(f"[cyan]Processed {idx + 1}/{total_records} records[/cyan]")
    
    new_memory.persist()
    
    print(f"\n[green]Compaction complete![/green]")
    print(f"[green]Original database: {total_records} records with non-continuous IDs[/green]")
    print(f"[green]Compacted database: {total_records} records with continuous IDs[/green]")
    
    # Show comparison
    print(f"\n[yellow]ID Mapping Examples:[/yellow]")
    result = new_memory.scenario_memory._collection.get(include=['ids'], limit=5)
    for i, new_id in enumerate(result['ids']):
        print(f"  Position {i+1}: {new_id}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Database Compaction Tool - Re-index ChromaDB with continuous IDs',
        epilog="""
Examples:
  # Compact default database
  python compact_database.py
  
  # Compact custom database
  python compact_database.py --db-path ./db/custom_db/
  
  # Compact and save to specific path
  python compact_database.py --db-path ./db/old_db/ --output ./db/new_db/
        """
    )
    
    parser.add_argument('--db-path', default='./db/chroma_5_shot_20_mem/',
                        help='Source database path (default: ./db/chroma_5_shot_20_mem/)')
    parser.add_argument('--output', help='Output database path (default: <db-path>_compacted/)')
    parser.add_argument('--force', action='store_true', 
                        help='Force compaction without confirmation')
    
    args = parser.parse_args()
    
    if not args.force:
        print("[yellow]Warning: This will create a NEW database with continuous IDs[/yellow]")
        print("[yellow]The original database will NOT be modified[/yellow]")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    
    compact_database(args.db_path, args.output)


if __name__ == '__main__':
    main()
