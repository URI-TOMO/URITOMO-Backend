import os
import redis
from rq import Worker, Queue, Connection
from app.core.config import settings

# Redis connection for RQ
redis_conn = redis.from_url(settings.redis_url)

# Queues to listen to
listen = ["default", "high", "low"]

def run_worker():
    """Start the RQ worker"""
    with Connection(redis_conn):
        worker = Worker(map(Queue, listen))
        worker.work()

if __name__ == "__main__":
    print(f"Starting worker listening on: {listen}")
    run_worker()
