Below is the **English translation** of your document.

---

# Example CRUD API

This directory contains a simple CRUD API example based on **SQLAlchemy models**.

## 📁 Structure

```
app/example/
├── __init__.py       # Package initialization
├── schemas.py        # Pydantic schemas (request/response models)
├── crud.py           # CRUD service logic
└── router.py         # FastAPI router (API endpoints)
```

## 🎯 Included Features

### User CRUD

* ✅ **POST** `/api/v1/example/users` – Create a user
* ✅ **GET** `/api/v1/example/users/{user_id}` – Retrieve a user
* ✅ **GET** `/api/v1/example/users` – Retrieve all users (pagination)
* ✅ **PATCH** `/api/v1/example/users/{user_id}` – Update user information
* ✅ **DELETE** `/api/v1/example/users/{user_id}` – Delete a user

### Room CRUD

* ✅ **POST** `/api/v1/example/rooms` – Create a room
* ✅ **GET** `/api/v1/example/rooms/{room_id}` – Retrieve a room
* ✅ **GET** `/api/v1/example/rooms` – Retrieve all rooms (pagination, creator filter)
* ✅ **PATCH** `/api/v1/example/rooms/{room_id}` – Update room information
* ✅ **DELETE** `/api/v1/example/rooms/{room_id}` – Delete a room (soft delete / hard delete)

## 🚀 How to Use

### 1. Run with Docker

```bash
# Start all services
docker-compose up -d

# Run database migrations
docker-compose exec api alembic upgrade head
```

### 2. Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="mysql+aiomysql://user:pass@localhost:3306/uritomo"
export REDIS_URL="redis://localhost:6379/0"
export QDRANT_URL="http://localhost:6333"

# Run database migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

### 3. Test with Swagger UI

After starting the server, open in your browser:

* **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 📝 API Usage Examples

### Create User

```bash
curl -X POST "http://localhost:8000/api/v1/example/users" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "user_001",
    "email": "test@example.com",
    "display_name": "Test User",
    "locale": "ko",
    "status": "active"
  }'
```

### Get User

```bash
curl -X GET "http://localhost:8000/api/v1/example/users/user_001"
```

### Get User List (Pagination)

```bash
curl -X GET "http://localhost:8000/api/v1/example/users?skip=0&limit=10"
```

### Update User

```bash
curl -X PATCH "http://localhost:8000/api/v1/example/users/user_001" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "Updated Name",
    "locale": "en"
  }'
```

### Delete User

```bash
curl -X DELETE "http://localhost:8000/api/v1/example/users/user_001"
```

### Create Room

```bash
curl -X POST "http://localhost:8000/api/v1/example/rooms" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "room_001",
    "title": "Test Room",
    "created_by": "user_001",
    "status": "active"
  }'
```

### Get Room

```bash
curl -X GET "http://localhost:8000/api/v1/example/rooms/room_001"
```

### Get Room List (Filter by Creator)

```bash
curl -X GET "http://localhost:8000/api/v1/example/rooms?creator_id=user_001&skip=0&limit=10"
```

### Delete Room (Soft Delete)

```bash
curl -X DELETE "http://localhost:8000/api/v1/example/rooms/room_001"
```

### Delete Room (Hard Delete)

```bash
curl -X DELETE "http://localhost:8000/api/v1/example/rooms/room_001?hard=true"
```

## 🔍 Key Features

### 1. **Pydantic Schema Validation**

* Automatic request data validation
* Guaranteed type safety
* Automatic API documentation generation

### 2. **Asynchronous Database Operations**

* Uses SQLAlchemy Async
* High performance and scalability

### 3. **Error Handling**

* 404 Not Found
* 409 Conflict (duplicate data)
* 422 Validation Error

### 4. **Pagination**

* Supports `skip` / `limit` parameters
* Efficient handling of large datasets

### 5. **Soft Delete**

* Sets an `ended_at` timestamp when deleting a room
* Allows data recovery

## 🏗️ Architecture Pattern

This example follows a **Layered Architecture** pattern:

```
┌─────────────────────────────────┐
│   API Layer (router.py)         │  ← FastAPI endpoints
├─────────────────────────────────┤
│   Schema Layer (schemas.py)     │  ← Pydantic models
├─────────────────────────────────┤
│   Service Layer (crud.py)       │  ← Business logic
├─────────────────────────────────┤
│   Model Layer (app/models/)     │  ← SQLAlchemy models
└─────────────────────────────────┘
```

## 📚 References

* FastAPI Official Documentation
* SQLAlchemy Official Documentation
* Pydantic Official Documentation

## 🎓 Learning Points

1. **RESTful API Design**: Proper use of HTTP methods and status codes
2. **Dependency Injection**: Database session management using FastAPI’s `Depends`
3. **Asynchronous Programming**: async/await patterns
4. **Data Validation**: Type safety with Pydantic
5. **ORM Usage**: Database abstraction with SQLAlchemy

---

**Note**: This example is intended for learning and testing purposes. In a production environment, additional features such as authentication, authorization, logging, and monitoring are required.
