#!/usr/bin/env python3
"""
Generate Alembic migration for the current models.
This script creates a new migration file based on the current state of SQLAlchemy models.
"""

import subprocess
import sys
from datetime import datetime


def main():
    """Generate a new Alembic migration"""
    
    # Generate timestamp-based migration message
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    message = f"create_initial_schema_{timestamp}"
    
    print("🔄 Generating Alembic migration...")
    print(f"📝 Migration message: {message}")
    
    try:
        # Run alembic revision command
        result = subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", message],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Migration generated successfully!")
        print(result.stdout)
        
        print("\n" + "="*60)
        print("Next steps:")
        print("1. Review the generated migration file in migrations/versions/")
        print("2. Apply the migration: alembic upgrade head")
        print("="*60)
        
    except subprocess.CalledProcessError as e:
        print("❌ Failed to generate migration!")
        print(f"Error: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Alembic not found!")
        print("Please install alembic: pip install alembic")
        sys.exit(1)


if __name__ == "__main__":
    main()
