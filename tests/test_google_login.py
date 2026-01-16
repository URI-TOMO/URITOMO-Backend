"""
Test suite for Google Login System

Tests the authentication flow, user creation, and token management.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.user.google_login import (
    GoogleAuthService,
    GoogleLoginCRUD,
    GoogleTokenRequest,
    GoogleUserInfo,
    google_login,
)
from app.models.user import User


# ============ Test Fixtures ============

@pytest.fixture
def mock_google_user_info():
    """Mock Google user information"""
    return GoogleUserInfo(
        sub="google_user_123",
        email="test@example.com",
        name="Test User",
        picture="https://example.com/photo.jpg",
        locale="ja"
    )


@pytest.fixture
def mock_user():
    """Mock database user"""
    return User(
        id="google_user_123",
        email="test@example.com",
        display_name="Test User",
        locale="ja",
        status="active",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


# ============ Unit Tests ============

class TestGoogleAuthService:
    """Test Google authentication service"""

    @patch('app.user.google_login.id_token.verify_oauth2_token')
    @patch('app.user.google_login.settings')
    def test_verify_google_token_success(self, mock_settings: Any, mock_verify: Any) -> None:
        """Test successful Google token verification"""
        # Setup
        mock_settings.google_client_id = "test_client_id"
        mock_verify.return_value = {
            'iss': 'accounts.google.com',
            'sub': 'google_user_123',
            'email': 'test@example.com',
            'name': 'Test User',
            'picture': 'https://example.com/photo.jpg',
            'locale': 'ja'
        }

        # Execute
        result = GoogleAuthService.verify_google_token("mock_token")

        # Verify
        assert result.sub == "google_user_123"
        assert result.email == "test@example.com"
        assert result.name == "Test User"
        assert result.locale == "ja"

    @patch('app.user.google_login.id_token.verify_oauth2_token')
    @patch('app.user.google_login.settings')
    def test_verify_google_token_no_client_id(self, mock_settings: Any, mock_verify: Any) -> None:
        """Test token verification fails without client ID"""
        # Setup
        mock_settings.google_client_id = None

        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            GoogleAuthService.verify_google_token("mock_token")
        
        assert "GOOGLE_CLIENT_ID not configured" in str(exc_info.value)

    @patch('app.user.google_login.id_token.verify_oauth2_token')
    @patch('app.user.google_login.settings')
    def test_verify_google_token_wrong_issuer(self, mock_settings: Any, mock_verify: Any) -> None:
        """Test token verification fails with wrong issuer"""
        # Setup
        mock_settings.google_client_id = "test_client_id"
        mock_verify.return_value = {
            'iss': 'malicious.com',  # Wrong issuer
            'sub': 'google_user_123',
            'email': 'test@example.com',
        }

        # Execute & Verify
        with pytest.raises(Exception) as exc_info:
            GoogleAuthService.verify_google_token("mock_token")
        
        assert "Invalid Google token" in str(exc_info.value)


class TestGoogleLoginCRUD:
    """Test CRUD operations for Google login"""

    @pytest.mark.asyncio
    async def test_get_or_create_user_new_user(self, mock_google_user_info: GoogleUserInfo) -> None:
        """Test creating a new user"""
        # Setup mock database session
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # User doesn't exist
        mock_db.execute.return_value = mock_result

        # Execute
        user = await GoogleLoginCRUD.get_or_create_user(mock_db, mock_google_user_info)

        # Verify
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        assert user.id == mock_google_user_info.sub
        assert user.email == mock_google_user_info.email

    @pytest.mark.asyncio
    async def test_get_or_create_user_existing_user(self, mock_google_user_info: GoogleUserInfo, mock_user: User) -> None:
        """Test getting existing user"""
        # Setup mock database session
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result

        # Execute
        user = await GoogleLoginCRUD.get_or_create_user(mock_db, mock_google_user_info)

        # Verify
        mock_db.add.assert_not_called()  # Should not add new user
        assert user.id == mock_user.id


# ============ Integration Tests ============

class TestGoogleLoginEndpoints:
    """Test Google login API endpoints"""

    @pytest.mark.asyncio
    @patch('app.user.google_login.GoogleAuthService.verify_google_token')
    @patch('app.user.google_login.GoogleLoginCRUD.get_or_create_user')
    @patch('app.user.google_login.create_access_token')
    async def test_google_login_success(
        self, 
        mock_create_token: Any,
        mock_get_or_create_user: Any,
        mock_verify_token: Any,
        mock_google_user_info: GoogleUserInfo,
        mock_user: User
    ) -> None:
        """Test successful Google login"""
        # Setup
        mock_verify_token.return_value = mock_google_user_info
        mock_get_or_create_user.return_value = mock_user
        mock_create_token.return_value = "mock_jwt_token"

        mock_db = MagicMock(spec=AsyncSession)
        token_request = GoogleTokenRequest(token="mock_google_token")

        # Execute
        result = await google_login(token_request, mock_db)

        # Verify
        assert result.access_token == "mock_jwt_token"
        assert result.token_type == "bearer"
        assert result.user.id == mock_user.id
        assert mock_verify_token.called
        assert mock_get_or_create_user.called
        assert mock_create_token.called


# ============ Schema Validation Tests ============

class TestSchemas:
    """Test Pydantic schemas"""

    def test_google_token_request_valid(self) -> None:
        """Test valid Google token request"""
        data = {"token": "valid_token_string"}
        request = GoogleTokenRequest(**data)
        assert request.token == "valid_token_string"

    def test_google_user_info_valid(self) -> None:
        """Test valid Google user info"""
        data = {
            "sub": "123",
            "email": "test@example.com",
            "name": "Test User",
            "locale": "ja"
        }
        user_info = GoogleUserInfo(**data)
        assert user_info.sub == "123"
        assert user_info.email == "test@example.com"

    def test_google_user_info_invalid_email(self) -> None:
        """Test invalid email in Google user info"""
        with pytest.raises(Exception):
            GoogleUserInfo(
                sub="123",
                email="invalid-email",  # Invalid email format
                name="Test User"
            )


# ============ Manual Testing Guide ============

def print_manual_test_guide() -> None:
    """
    Print guide for manual testing with real Google OAuth
    """
    guide = """
    ====================================
    Google Login Manual Testing Guide
    ====================================
    
    1. Setup Google OAuth:
       - Go to Google Cloud Console
       - Create OAuth 2.0 Client ID
       - Add to .env:
         GOOGLE_CLIENT_ID=your_client_id
         GOOGLE_CLIENT_SECRET=your_client_secret
    
    2. Frontend Integration:
       - Use Google Sign-In button
       - Get ID token from Google
       - Send to: POST /api/v1/auth/google
         Body: {"token": "google_id_token"}
    
    3. Test Endpoints:
       a) Login: POST /auth/google
          Request: {"token": "<google_token>"}
          Response: {"access_token": "...", "user": {...}}
       
       b) Get User: GET /auth/me
          Header: Authorization: Bearer <jwt_token>
          Response: {"id": "...", "email": "...", ...}
       
       c) Refresh: POST /auth/refresh
          Header: Authorization: Bearer <jwt_token>
          Response: {"access_token": "...", "user": {...}}
    
    4. Expected Flow:
       - User clicks "Sign in with Google"
       - Google returns ID token
       - Backend verifies token
       - Backend creates/updates user
       - Backend returns JWT token
       - Frontend stores JWT token
       - Frontend uses JWT for authenticated requests
    """
    print(guide)


if __name__ == "__main__":
    print_manual_test_guide()
    print("\n? Running pytest...")
    pytest.main([__file__, "-v"])
