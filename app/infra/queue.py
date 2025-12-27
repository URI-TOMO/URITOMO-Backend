"""
Queue infrastructure configuration

Redis Queue (RQ) abstraction for background jobs.
"""

from typing import Any, Dict, Optional

from redis import Redis
from rq import Queue
from rq.job import Job

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class JobQueue:
    """Wrapper around RQ Queue"""
    
    def __init__(self, redis_conn: Redis, queue_name: str = "default"):
        self.redis = redis_conn
        self.queue_name = queue_name
        self.queue = Queue(queue_name, connection=self.redis)

    def enqueue(
        self,
        func: str,  # "module.path.to.func"
        args: Optional[tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = settings.worker_job_timeout,
        result_ttl: int = settings.worker_result_ttl,
        job_id: Optional[str] = None,
    ) -> Job:
        """Enqueue a job"""
        try:
            job = self.queue.enqueue(
                func,
                args=args,
                kwargs=kwargs,
                result_ttl=result_ttl,
                job_timeout=timeout,
                job_id=job_id,
            )
            logger.info(
                f"Job enqueued: {job.id}",
                function=func,
                queue=self.queue_name,
            )
            return job
        except Exception as e:
            logger.error(f"Failed to enqueue job: {e}")
            raise

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        try:
            return Job.fetch(job_id, connection=self.redis)
        except Exception:
            return None


class QueueFactory:
    """Factory to get queue instances"""
    
    _queues: Dict[str, JobQueue] = {}
    
    @classmethod
    def get_queue(cls, redis_conn: Redis, name: str = "default") -> JobQueue:
        if name not in cls._queues:
            cls._queues[name] = JobQueue(redis_conn, name)
        return cls._queues[name]
