#!/usr/bin/env python3
"""
Vector Database Management Tool
Provides CLI commands to manage ChromaDB memory records
"""

import argparse
import sys
import json
from lsda.driver_agent.vectorStore import DrivingMemory


def print_record(record, index=None):
    """Pretty print a memory record"""
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n{prefix}ID: {record['id']}")
    print(f"Content: {record['content'][:100]}..." if len(record['content']) > 100 else f"Content: {record['content']}")
    print(f"Metadata: {json.dumps(record['metadata'], indent=2)}")


def cmd_list(args):
    """List all records in database"""
    memory = DrivingMemory(db_path=args.db_path)
    
    if args.comment:
        print(f"[cyan]Filtering records with comment: {args.comment}[/cyan]")
    
    records = memory.getMemory(limit=args.limit)
    
    if not records:
        print("[yellow]No records found[/yellow]")
        return
    
    filtered_records = records
    if args.comment:
        filtered_records = [r for r in records if r['metadata'].get('comments') == args.comment]
    
    print(f"\n[green]Total: {len(filtered_records)} records[/green]")
    
    for i, record in enumerate(filtered_records[:args.limit]):
        print_record(record, i + 1)


def cmd_get(args):
    """Get a specific record by ID"""
    memory = DrivingMemory(db_path=args.db_path)
    
    record = memory.getMemory(record_id=args.id)
    if record:
        print_record(record)
    else:
        print(f"[red]Record {args.id} not found[/red]")


def cmd_edit(args):
    """Edit a specific record"""
    memory = DrivingMemory(db_path=args.db_path)
    
    print(f"[cyan]Editing record {args.id}...[/cyan]")
    
    memory.editMemory(
        record_id=args.id,
        human_question=args.question,
        response=args.response,
        action=int(args.action) if args.action else None,
        comments=args.comments
    )
    memory.persist()


def cmd_delete(args):
    """Delete records"""
    memory = DrivingMemory(db_path=args.db_path)
    
    if args.id:
        print(f"[cyan]Deleting record {args.id}...[/cyan]")
        memory.deleteMemory([args.id])
    elif args.comment:
        print(f"[cyan]Deleting all records with comment: {args.comment}[/cyan]")
        memory.deleteByComment(args.comment)
    else:
        print("[red]Error: Must specify either --id or --comment[/red]")
        return
    
    memory.persist()


def cmd_stats(args):
    """Show database statistics"""
    memory = DrivingMemory(db_path=args.db_path)
    stats = memory.getStatistics()


def cmd_export(args):
    """Export records to JSON"""
    memory = DrivingMemory(db_path=args.db_path)
    
    records = memory.getMemory(limit=args.limit)
    if not records:
        print("[yellow]No records to export[/yellow]")
        return
    
    export_data = {
        'total': len(records),
        'records': records
    }
    
    with open(args.output, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"[green]Exported {len(records)} records to {args.output}[/green]")


def cmd_clean(args):
    """Clean database - remove records by comment"""
    memory = DrivingMemory(db_path=args.db_path)
    
    comments_to_remove = args.comments.split(',')
    total_deleted = 0
    
    for comment in comments_to_remove:
        comment = comment.strip()
        print(f"[cyan]Removing records with comment: {comment}[/cyan]")
        result = memory.scenario_memory._collection.get(
            where={"comments": {"$eq": comment}},
            include=['metadatas']
        )
        if len(result['ids']) > 0:
            memory.deleteByComment(comment)
            total_deleted += len(result['ids'])
    
    memory.persist()
    print(f"[green]Total deleted: {total_deleted} records[/green]")


def main():
    parser = argparse.ArgumentParser(
        description='Vector Database Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all records
  python manage_vector_db.py list
  
  # List records with specific comment
  python manage_vector_db.py list --comment collision-reflection
  
  # Get a specific record
  python manage_vector_db.py get --id <record_id>
  
  # Edit a record
  python manage_vector_db.py edit --id <record_id> --response "新的响应" --comments "已修改"
  
  # Delete a record by ID
  python manage_vector_db.py delete --id <record_id>
  
  # Delete all records with specific comment
  python manage_vector_db.py delete --comment collision-reflection
  
  # Show database statistics
  python manage_vector_db.py stats
  
  # Export records to JSON
  python manage_vector_db.py export --output backup.json
  
  # Clean database - remove multiple comment types
  python manage_vector_db.py clean --comments "collision-reflection,old-record"
        """
    )
    
    parser.add_argument('--db-path', default='./db/chroma_5_shot_20_mem/',
                        help='Path to ChromaDB database (default: ./db/chroma_5_shot_20_mem/)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all records')
    list_parser.add_argument('--limit', type=int, default=1000, help='Limit number of records (default: 1000)')
    list_parser.add_argument('--comment', help='Filter by comment')
    list_parser.set_defaults(func=cmd_list)
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get a specific record')
    get_parser.add_argument('--id', required=True, help='Record ID')
    get_parser.set_defaults(func=cmd_get)
    
    # Edit command
    edit_parser = subparsers.add_parser('edit', help='Edit a record')
    edit_parser.add_argument('--id', required=True, help='Record ID')
    edit_parser.add_argument('--question', help='Update human question')
    edit_parser.add_argument('--response', help='Update LLM response')
    edit_parser.add_argument('--action', help='Update action (0-4)')
    edit_parser.add_argument('--comments', help='Update comments')
    edit_parser.set_defaults(func=cmd_edit)
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete records')
    delete_parser.add_argument('--id', help='Record ID to delete')
    delete_parser.add_argument('--comment', help='Delete all records with this comment')
    delete_parser.set_defaults(func=cmd_delete)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export records to JSON')
    export_parser.add_argument('--output', default='db_export.json', help='Output file (default: db_export.json)')
    export_parser.add_argument('--limit', type=int, default=1000, help='Limit records to export (default: 1000)')
    export_parser.set_defaults(func=cmd_export)
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean database by removing specific comments')
    clean_parser.add_argument('--comments', required=True, 
                              help='Comma-separated list of comments to remove (e.g., "collision-reflection,old-record")')
    clean_parser.set_defaults(func=cmd_clean)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == '__main__':
    main()
