#!/usr/bin/env python3
"""
Memory Management Tool for L-SDA
Provides utilities to view, export, and manage the reflection memory database
"""

import os
import yaml
import json
import shutil
from datetime import datetime
from pathlib import Path
from lsda.driver_agent.vectorStore import DrivingMemory

# Set environment variables from config
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
os.environ["OPENAI_API_TYPE"] = 'openai'
os.environ["OPENAI_API_KEY"] = config['gpt_config']['OPENAI_KEY']
os.environ["OPENAI_CHAT_MODEL"] = config['gpt_config']['OPENAI_CHAT_MODEL']
os.environ["OPENAI_API_BASE"] = config['gpt_config']['OPENAI_API_BASE']


def view_memory(db_path=None):
    """View all memory entries in the database"""
    if db_path is None:
        db_path = config.get('memory_path', 'memories/testmemo')
    print(f"\n{'='*80}")
    print(f"[MEMORY DATABASE VIEW] - {db_path}")
    print(f"{'='*80}\n")
    
    try:
        memory = DrivingMemory(db_path=db_path)
        all_data = memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        total_items = len(all_data['ids'])
        print(f"Total memory items: {total_items}\n")
        
        if total_items == 0:
            print("[INFO] Database is empty")
            return
        
        action_names = {0: "Turn-left", 1: "IDLE", 2: "Turn-right", 3: "Acceleration", 4: "Deceleration"}
        
        for i, (doc_id, document, metadata) in enumerate(zip(
            all_data['ids'], 
            all_data['documents'], 
            all_data['metadatas']
        ), 1):
            print(f"\n[Record {i}] ID: {doc_id}")
            print(f"{'─'*80}")
            action_name = action_names.get(metadata['action'], f"UNKNOWN_{metadata['action']}")
            print(f"Action: {metadata['action']} ({action_name})")
            print(f"Comments: {metadata.get('comments', 'N/A')}")
            print(f"\nScenario Description:")
            print(document)
            print(f"\nLLM Response:")
            print(metadata['LLM_response'])
            print(f"{'─'*80}")
    
    except Exception as e:
        print(f"[ERROR] Failed to view memory: {e}")


def export_memory(db_path=None, output_dir="memory_export"):
    """Export memory to files (JSON, TXT, statistics)"""
    if db_path is None:
        db_path = config.get('memory_path', 'memories/testmemo')
    print(f"\n[EXPORTING MEMORY] from {db_path}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        memory = DrivingMemory(db_path=db_path)
        all_data = memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        memories_data = []
        for i, (doc_id, document, metadata) in enumerate(zip(
            all_data['ids'], 
            all_data['documents'], 
            all_data['metadatas']
        ), 1):
            memory_record = {
                "record_id": i,
                "database_id": doc_id,
                "scenario_description": document,
                "action": metadata['action'],
                "llm_response": metadata['LLM_response'],
                "comments": metadata.get('comments', 'N/A')
            }
            memories_data.append(memory_record)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save as JSON
        json_file = os.path.join(output_dir, f"memories_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(memories_data, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON exported: {json_file}")
        
        # 2. Save as readable TXT
        txt_file = os.path.join(output_dir, f"memories_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Memory Database Export\n")
            f.write(f"Database: {db_path}\n")
            f.write(f"Export time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total records: {len(memories_data)}\n")
            f.write("="*80 + "\n\n")
            
            for record in memories_data:
                f.write(f"Record {record['record_id']}:\n")
                f.write(f"  Action: {record['action']}\n")
                f.write(f"  Comments: {record['comments']}\n")
                f.write(f"  Scenario: {record['scenario_description']}\n")
                f.write(f"  Response: {record['llm_response']}\n")
                f.write("-"*80 + "\n\n")
        print(f"✅ Text exported: {txt_file}")
        
        # 3. Save statistics
        stats_file = os.path.join(output_dir, f"memory_stats_{timestamp}.txt")
        action_counts = {}
        comment_counts = {}
        action_names = {0: "Turn-left", 1: "IDLE", 2: "Turn-right", 3: "Acceleration", 4: "Deceleration"}
        
        for record in memories_data:
            action = record['action']
            comment = record['comments']
            action_counts[action] = action_counts.get(action, 0) + 1
            comment_counts[comment] = comment_counts.get(comment, 0) + 1
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"Memory Statistics\n")
            f.write(f"Database: {db_path}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total records: {len(memories_data)}\n\n")
            
            f.write("Action Distribution:\n")
            for action_id, count in sorted(action_counts.items()):
                action_name = action_names.get(action_id, f"UNKNOWN_{action_id}")
                percentage = (count / len(memories_data)) * 100
                f.write(f"  Action {action_id} ({action_name}): {count} ({percentage:.1f}%)\n")
            
            f.write("\nComment Distribution:\n")
            for comment, count in sorted(comment_counts.items()):
                percentage = (count / len(memories_data)) * 100
                f.write(f"  '{comment}': {count} ({percentage:.1f}%)\n")
        
        print(f"✅ Statistics exported: {stats_file}")
        print(f"\n[SUCCESS] Export completed. Total records: {len(memories_data)}")
        
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")


def clear_memory(db_path=None, confirm=True):
    """Clear all items from memory database"""
    if db_path is None:
        db_path = config.get('memory_path', 'memories/testmemo')
    if confirm:
        response = input(f"\n⚠️ Are you sure you want to clear all items from {db_path}? (Y/N): ").strip().upper()
        if response != 'Y':
            print("[CANCELLED] Clear operation cancelled")
            return
    
    try:
        # Delete and recreate directory
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)
        print(f"[SUCCESS] Memory cleared: {db_path}")
    except Exception as e:
        print(f"[ERROR] Failed to clear memory: {e}")


def show_memory_stats():
    """Show memory database statistics"""
    print(f"\n{'='*80}")
    print(f"[MEMORY DATABASE STATISTICS]")
    print(f"{'='*80}\n")
    
    db_path = config.get('memory_path', 'memories/testmemo')
    if not os.path.exists(db_path):
        print(f"[INFO] Database not found: {db_path}")
        return
    
    try:
        memory = DrivingMemory(db_path=db_path)
        all_data = memory.scenario_memory._collection.get(
            include=['documents', 'metadatas']
        )
        
        total = len(all_data['ids'])
        action_names = {0: "Turn-left", 1: "IDLE", 2: "Turn-right", 3: "Acceleration", 4: "Deceleration"}
        
        action_dist = {}
        comment_dist = {}
        
        for metadata in all_data['metadatas']:
            action = metadata['action']
            comment = metadata.get('comments', 'N/A')
            action_dist[action] = action_dist.get(action, 0) + 1
            comment_dist[comment] = comment_dist.get(comment, 0) + 1
        
        print(f"Total memory items: {total}\n")
        
        if total > 0:
            print("Action Distribution:")
            for action_id in sorted(action_dist.keys()):
                action_name = action_names.get(action_id, f"UNKNOWN_{action_id}")
                count = action_dist[action_id]
                percentage = (count / total) * 100
                print(f"  • {action_name}: {count} ({percentage:.1f}%)")
            
            print("\nComment Distribution:")
            for comment, count in sorted(comment_dist.items()):
                percentage = (count / total) * 100
                print(f"  • {comment}: {count} ({percentage:.1f}%)")
        else:
            print("[INFO] No items in database")
        
        print(f"\n{'='*80}")
    
    except Exception as e:
        print(f"[ERROR] Failed to get statistics: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("""
L-SDA Memory Management Tool

Usage:
  python manage_memories.py view      - View all memory entries
  python manage_memories.py export    - Export memory to files
  python manage_memories.py stats     - Show memory statistics
  python manage_memories.py clear     - Clear all memory entries

Examples:
  python manage_memories.py view
  python manage_memories.py export
  python manage_memories.py stats
        """)
    else:
        command = sys.argv[1].lower()
        if command == 'view':
            view_memory()
        elif command == 'export':
            export_memory()
        elif command == 'stats':
            show_memory_stats()
        elif command == 'clear':
            clear_memory()
        else:
            print(f"[ERROR] Unknown command: {command}")
