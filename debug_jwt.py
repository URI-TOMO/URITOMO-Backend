"""
Debug JWT token verification
"""
import jwt
import json

# Exact key from .env
SECRET_KEY = "dev-secret-key-for-testing-0123456789abcdefghij"
ALGORITHM = "HS256"

# Token from test
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhY2NvdW50cy5nb29nbGUuY29tIiwic3ViIjoiZ29vZ2xlX3VzZXJfMTIzIiwiZW1haWwiOiJ1c2VyMUBleGFtcGxlLmNvbSIsIm5hbWUiOiJUZXN0IFVzZXIgMSIsInBpY3R1cmUiOiJodHRwczovL2V4YW1wbGUuY29tL3Bob3RvLmpwZyIsImxvY2FsZSI6ImphIiwiYXVkIjoidGVzdF9jbGllbnRfaWQiLCJleHAiOjE3NjkwMjAxMjEsImlhdCI6MTc2OTAxNjUyMX0.-u8NxIa6GelMJGpIIyyfbktPVQ0sjy-7WEGBB4r_nr8"

print("[DEBUG] Token Verification Test")
print(f"[DEBUG] SECRET_KEY: {SECRET_KEY}")
print(f"[DEBUG] SECRET_KEY length: {len(SECRET_KEY)}")
print(f"[DEBUG] ALGORITHM: {ALGORITHM}")
print(f"[DEBUG] Token (first 50 chars): {TOKEN[:50]}...")

# Try to decode
try:
    payload = jwt.decode(
        TOKEN,
        SECRET_KEY,
        algorithms=[ALGORITHM]
    )
    print("\n[SUCCESS] Token decoded successfully!")
    print(f"[DEBUG] Payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"\n[ERROR] Token decode failed!")
    print(f"[ERROR] Error type: {type(e).__name__}")
    print(f"[ERROR] Error message: {str(e)}")
    import traceback
    traceback.print_exc()

# Try decoding without verification to see the payload
print("\n[DEBUG] Decoding without verification to inspect payload:")
try:
    unverified = jwt.decode(TOKEN, options={"verify_signature": False})
    print(f"[DEBUG] Unverified payload:")
    print(json.dumps(unverified, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"[ERROR] Even unverified decode failed: {e}")
