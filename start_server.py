"""
Start the URITOMO backend server for testing
"""
import uvicorn
import sys

if __name__ == "__main__":
    print("Starting URITOMO Backend Server...")
    print(f"Python: {sys.version}")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        raise
