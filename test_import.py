"""Simple import test for google_login module"""
import sys
sys.path.insert(0, ".")

try:
    from app.user import google_login
    print("? Import successful!")
    print(f"Router prefix: {google_login.router.prefix}")
    print(f"Router tags: {google_login.router.tags}")
    print(f"Number of endpoints: {len(google_login.router.routes)}")
    print("\nEndpoints:")
    for route in google_login.router.routes:
        methods = list(route.methods) if hasattr(route, 'methods') else []
        print(f"  - {methods} {route.path}")
    print("\n? Google Login system is ready!")
except ImportError as e:
    print(f"? Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"? Error: {e}")
    sys.exit(1)
