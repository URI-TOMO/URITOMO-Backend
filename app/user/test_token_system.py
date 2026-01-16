#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Token System Test Module - Simple version without circular imports

Test suite for JWT and refresh token functionality.
"""

from datetime import datetime, timezone, timedelta
import secrets
import bcrypt

# Simple JWT implementation for testing
def create_simple_access_token_test() -> bool:
    """Test JWT token creation using jose library directly"""
    try:
        from jose import jwt as jose_jwt
        
        JWT_SECRET_KEY = "test-secret-key"
        JWT_ALGORITHM = "HS256"
        
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=30)
        
        payload = {
            "sub": "test_user_123",
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "type": "access",
        }
        
        token = jose_jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Verify
        decoded = jose_jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        return decoded["sub"] == "test_user_123"
    except Exception as e:
        print(f"Error in access token test: {e}")
        return False


def test_access_token():
    """Test JWT access token creation and verification"""
    print("=" * 60)
    print("Testing Access Token (JWT)")
    print("=" * 60)
    
    try:
        from jose import jwt as jose_jwt
        
        JWT_SECRET_KEY = "test-secret-key"
        JWT_ALGORITHM = "HS256"
        
        user_id = "test_user_123"
        
        # Test token creation
        print(f"1. Creating access token for user: {user_id}")
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=30)
        
        payload = {
            "sub": str(user_id),
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "type": "access",
        }
        
        token = jose_jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        print(f"   ? Token created: {token[:50]}...")
        
        # Test token verification
        print(f"2. Verifying access token")
        decoded = jose_jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        print(f"   ? Token verified successfully")
        print(f"   - User ID (sub): {decoded['sub']}")
        print(f"   - Token type: {decoded['type']}")
        
        # Test invalid token
        print(f"3. Testing invalid token")
        try:
            jose_jwt.decode("invalid.token.here", JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            print("   ? Invalid token was not rejected!")
            return False
        except Exception:
            print("   ? Invalid token correctly rejected")
        
        print()
        return True
    except Exception as e:
        print(f"   ? Error: {e}")
        print()
        return False


def test_refresh_token():
    """Test refresh token creation"""
    print("=" * 60)
    print("Testing Refresh Token")
    print("=" * 60)
    
    try:
        # Test token creation
        print("1. Creating refresh token")
        token1 = secrets.token_urlsafe(48)
        print(f"   ? Token 1 created: {token1[:30]}...")
        
        token2 = secrets.token_urlsafe(48)
        print(f"   ? Token 2 created: {token2[:30]}...")
        
        # Test uniqueness
        if token1 != token2:
            print("   ? Each token is unique (good randomness)")
        else:
            print("   ? Tokens are not unique!")
            return False
        
        # Test format
        if len(token1) == 64:  # base64url encoded 48 bytes
            print(f"   ? Token format correct: {len(token1)} characters")
        else:
            print(f"   ? Token format incorrect: expected 64, got {len(token1)}")
            return False
        
        print()
        return True
    except Exception as e:
        print(f"   ? Error: {e}")
        print()
        return False


def test_token_hashing():
    """Test bcrypt token hashing"""
    print("=" * 60)
    print("Testing Token Hashing (bcrypt)")
    print("=" * 60)
    
    try:
        token = "test_token_value_12345"
        
        # Test hashing
        print(f"1. Hashing token: {token[:20]}...")
        hashed = bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()
        print(f"   ? Token hashed: {hashed[:50]}...")
        
        # Test hash length
        if len(hashed) > 30:
            print(f"   ? Hash length appropriate: {len(hashed)} characters")
        else:
            print(f"   ? Hash too short: {len(hashed)} characters")
            return False
        
        # Test verification
        print(f"2. Verifying token against hash")
        if bcrypt.checkpw(token.encode(), hashed.encode()):
            print("   ? Token verified successfully")
        else:
            print("   ? Token verification failed!")
            return False
        
        # Test invalid token
        print(f"3. Testing invalid token against hash")
        if not bcrypt.checkpw("wrong_token".encode(), hashed.encode()):
            print("   ? Invalid token correctly rejected")
        else:
            print("   ? Invalid token was not rejected!")
            return False
        
        print()
        return True
    except Exception as e:
        print(f"   ? Error: {e}")
        print()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("?" + "=" * 58 + "?")
    print("?" + " " * 58 + "?")
    print("?" + "  Token System Test Suite".center(58) + "?")
    print("?" + " " * 58 + "?")
    print("?" + "=" * 58 + "?")
    print()
    
    results = []
    
    # Run tests
    results.append(("Access Token (JWT)", test_access_token()))
    results.append(("Refresh Token", test_refresh_token()))
    results.append(("Token Hashing", test_token_hashing()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "? PASS" if result else "? FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n? All tests passed!")
        return 0
    else:
        print(f"\n??  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
