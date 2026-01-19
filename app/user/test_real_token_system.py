#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real Token System Test Module

Tests the actual logic in app.user.token_auth against real dependencies.
"""

import sys
import os
import asyncio
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.user.token_auth import (
    create_access_token,
    verify_access_token,
    create_refresh_token,
    hash_token,
    verify_token_hash,
    TokenPayload
)

def test_access_token_workflow():
    print("=" * 60)
    print("Testing Access Token Workflow (Real Implementation)")
    print("=" * 60)
    
    try:
        user_id = "test_real_user_123"
        print(f"1. Creating access token for user: {user_id}")
        token = create_access_token(user_id)
        print(f"   ? Token created: {token[:50]}...")
        
        print(f"2. Verifying access token")
        payload = verify_access_token(token)
        
        if payload:
            print(f"   ? Token verified successfully")
            print(f"   - User ID: {payload.sub}")
            print(f"   - Type: {payload.type}")
            
            if payload.sub == user_id:
                print("   ? User ID matches")
            else:
                print(f"   ? User ID Mismatch! Expected {user_id}, got {payload.sub}")
                return False
                
            if payload.type == "access":
                print("   ? Token type matches 'access'")
            else:
                print(f"   ? Token type Mismatch! Expected 'access', got {payload.type}")
                return False
        else:
            print("   ? Token verification failed (returned None)")
            return False
            
        return True
    except Exception as e:
        print(f"   ? Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_refresh_token_workflow():
    print("=" * 60)
    print("Testing Refresh Token Workflow (Real Implementation)")
    print("=" * 60)
    
    try:
        print("1. Creating refresh token")
        token = create_refresh_token()
        print(f"   ? Token created: {token[:50]}...")
        
        print("2. Hashing token")
        hashed = hash_token(token)
        print(f"   ? Hashed: {hashed[:50]}...")
        
        print("3. Verifying hash")
        is_valid = verify_token_hash(token, hashed)
        if is_valid:
            print("   ? Hash verified successfully")
        else:
            print("   ? Hash verification failed!")
            return False
            
        print("4. Verifying invalid token")
        is_invalid_valid = verify_token_hash("wrong_token", hashed)
        if not is_invalid_valid:
            print("   ? Invalid token correctly rejected")
        else:
            print("   ? Invalid token was accepted!")
            return False
            
        return True
    except Exception as e:
        print(f"   ? Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\nRunning REAL Token System Tests...\n")
    
    results = []
    results.append(("Access Token Workflow", test_access_token_workflow()))
    results.append(("Refresh Token Workflow", test_refresh_token_workflow()))
    
    print("\n" + "=" * 60)
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
