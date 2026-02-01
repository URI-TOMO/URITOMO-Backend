import sys
from redis import Redis
from rq import Connection, Worker, Queue

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

def run_worker() -> None:
    """Run RQ worker to process background jobs"""
    try:
        # Connect to Redis
        redis_url = settings.redis_url
        redis_conn = Redis.from_url(redis_url)
        
        # Create queues
        # Note: Worker queues are defined in settings.worker_queues (default: ["default", "high", "low"])
        queues = [Queue(name, connection=redis_conn) for name in settings.worker_queues]
        
        with Connection(redis_conn):
            worker = Worker(queues, connection=redis_conn)
            
            logger.info("Starting RQ worker", queues=settings.worker_queues)
            worker.work()
            
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_worker()
