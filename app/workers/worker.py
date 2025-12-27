"""
Worker Entry Point

Starts the Redis Queue (RQ) worker.
"""

import os
import sys

# Ensure app is in path
sys.path.append(os.getcwd())

from redis import Redis
from rq import Worker, Queue, Connection

from app.core.config import settings
from app.core.logging import setup_logging

listen = settings.worker_queues

if __name__ == "__main__":
    setup_logging()
    
    redis_url = settings.redis_url
    conn = Redis.from_url(redis_url)

    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        print(f"Worker started. Listening on: {listen}")
        worker.work()
