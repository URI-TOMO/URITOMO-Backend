import asyncio
import os
import sys
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.models.base import Base

async def debug_db():
    print(f"Connecting to database: {settings.database_url}")
    
    # Replace the driver if it's not async (though it should be in settings)
    db_url = settings.database_url
    if not db_url.startswith("mysql+aiomysql://"):
        db_url = db_url.replace("mysql://", "mysql+aiomysql://")
        db_url = db_url.replace("mysql+pymysql://", "mysql+aiomysql://")
    
    try:
        engine = create_async_engine(db_url)
        
        async with engine.connect() as conn:
            # Check connection
            result = await conn.execute(text("SELECT 1"))
            print("Connection successful!")
            
            # List tables
            result = await conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result]
            print(f"Tables in database: {tables}")
            
            # Check for specific tables
            expected_tables = [
                "users", "rooms", "room_members", "chat_messages", 
                "auth_tokens", "live", "ai_events", "user_friends",
                "dm_threads", "dm_participants", "dm_messages",
                "room_live_sessions", "room_live_session_members"
            ]
            
            print("\nTable Status:")
            for table in expected_tables:
                status = "✅ EXISTS" if table in tables else "❌ MISSING"
                print(f"  {table:30} {status}")
                
            # Print Alembic version
            try:
                result = await conn.execute(text("SELECT version_num FROM alembic_version"))
                version = result.scalar()
                print(f"\nCurrent Alembic Version: {version}")
            except Exception:
                print("\nAlembic version table not found.")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(debug_db())
