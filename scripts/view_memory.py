import os
import yaml
import json
from datetime import datetime
from lsda.driver_agent.vectorStore import DrivingMemory

# Set environment variables
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
if config['OPENAI_API_TYPE'] == 'azure':
    os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
    os.environ["OPENAI_API_VERSION"] = config['AZURE_API_VERSION']
    os.environ["OPENAI_API_BASE"] = config['AZURE_API_BASE']
    os.environ["OPENAI_API_KEY"] = config['AZURE_API_KEY']
    os.environ["AZURE_CHAT_DEPLOY_NAME"] = config['AZURE_CHAT_DEPLOY_NAME']
    os.environ["AZURE_EMBED_DEPLOY_NAME"] = config['AZURE_EMBED_DEPLOY_NAME']
elif config['OPENAI_API_TYPE'] == 'openai':
    os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
    os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
    os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
    os.environ["OPENAI_API_BASE"] = config['OPENAI_API_BASE']
else:
    raise ValueError("Unknown OPENAI_API_TYPE, should be azure or openai")

def save_memories_to_files(db_path="memories/20_mem", output_dir="memory_export"):
    """Save memory content to files"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DrivingMemory instance
    memory = DrivingMemory(db_path=db_path)
    
    # Get all data
    all_data = memory.scenario_memory._collection.get(
        include=['documents', 'metadatas', 'embeddings']
    )
    
    # Prepare data
    memories_data = []
    
    print(f"Total entries: {len(all_data['ids'])}")
    
    for i, (doc_id, document, metadata) in enumerate(zip(
        all_data['ids'], 
        all_data['documents'], 
        all_data['metadatas']
    )):
        memory_record = {
            "record_id": i + 1,
            "database_id": doc_id,
            "scenario_description": document,
            "human_question": metadata['human_question'],
            "llm_response": metadata['LLM_response'],
            "action": metadata['action'],
            "comments": metadata.get('comments', 'No comments')
        }
        memories_data.append(memory_record)
        
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save as JSON file
    json_filename = f"{output_dir}/memories_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(memories_data, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON file saved: {json_filename}")
    
    # 2. Save as readable text file
    txt_filename = f"{output_dir}/memories_{timestamp}.txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"Memory Database Export Report\n")
        f.write(f"Database path: {db_path}\n")
        f.write(f"Export time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total records: {len(memories_data)}\n")
        f.write("=" * 80 + "\n\n")
        
        for record in memories_data:
            f.write(f"Record {record['record_id']}:\n")
            f.write(f"  Database ID: {record['database_id']}\n")
            f.write(f"  Scenario description: {record['scenario_description']}\n")
            f.write(f"  Human question: {record['human_question']}\n")
            f.write(f"  LLM response: {record['llm_response']}\n")
            f.write(f"  Action: {record['action']}\n")
            f.write(f"  Comments: {record['comments']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"✅ Text file saved: {txt_filename}")
    
    # 3. Save statistics
    stats_filename = f"{output_dir}/memory_stats_{timestamp}.txt"
    action_counts = {}
    for record in memories_data:
        action = record['action']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    with open(stats_filename, 'w', encoding='utf-8') as f:
        f.write(f"Memory Database Statistics\n")
        f.write(f"Database path: {db_path}\n")
        f.write(f"Statistics time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total records: {len(memories_data)}\n\n")
        
        f.write("Action distribution:\n")
        action_names = {0: "Turn-left", 1: "IDLE", 2: "Turn-right", 3: "Acceleration", 4: "Deceleration"}
        for action_id, count in sorted(action_counts.items()):
            action_name = action_names.get(action_id, f"UNKNOWN_{action_id}")
            percentage = (count / len(memories_data)) * 100
            f.write(f"  Action {action_id} ({action_name}): {count} times ({percentage:.1f}%)\n")
        
        f.write(f"\nComments distribution:\n")
        comment_counts = {}
        for record in memories_data:
            comment = record['comments']
            comment_counts[comment] = comment_counts.get(comment, 0) + 1
        
        for comment, count in sorted(comment_counts.items()):
            percentage = (count / len(memories_data)) * 100
            f.write(f"  '{comment}': {count} times ({percentage:.1f}%)\n")
    
    print(f"✅ Statistics file saved: {stats_filename}")
    
    return json_filename, txt_filename, stats_filename

def save_specific_records(db_path="memories/20_mem", record_ids=None, output_dir="memory_export"):
    """Save specific records"""
    if record_ids is None:
        record_ids = [1, 2, 3]  # Default: save first 3 records
    
    os.makedirs(output_dir, exist_ok=True)
    memory = DrivingMemory(db_path=db_path)
    all_data = memory.scenario_memory._collection.get(include=['documents', 'metadatas'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/selected_memories_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Selected Records Export\n")
        f.write(f"Record IDs: {record_ids}\n")
        f.write(f"Export time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, record_id in enumerate(record_ids):
            if record_id <= len(all_data['ids']):
                idx = record_id - 1  # Convert to 0-based index
                f.write(f"Record {record_id}:\n")
                f.write(f"  Scenario description: {all_data['documents'][idx]}\n")
                f.write(f"  Action: {all_data['metadatas'][idx]['action']}\n")
                f.write(f"  LLM response: {all_data['metadatas'][idx]['LLM_response']}\n")
                f.write("-" * 80 + "\n\n")
    
    print(f"✅ Selected records saved: {filename}")
    return filename

if __name__ == "__main__":
    # Method 1: Save all memories to files
    json_file, txt_file, stats_file = save_memories_to_files()
    
    print(f"\nExport completed!")
    print(f"JSON file: {json_file}")
    print(f"Text file: {txt_file}")
    print(f"Statistics file: {stats_file}")
    
    # Method 2: Save only specific records (optional)
    # selected_file = save_specific_records(record_ids=[1, 2, 3, 4, 5])
    
    # If there is an updated database, it can also be exported
    if os.path.exists("memories/20_mem_updated"):
        print(f"\nFound updated memory database, exporting...")
        json_file2, txt_file2, stats_file2 = save_memories_to_files(
            db_path="memories/20_mem_updated", 
            output_dir="memory_export_updated"
        )
        print(f"✅ Updated version export completed: {txt_file2}")