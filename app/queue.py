from redis import Redis
from rq import Queue

from app.config.settings import settings


def get_redis_connection() -> Redis:
    return Redis.from_url(settings.redis_url)


def redis_is_healthy() -> bool:
    try:
        return bool(get_redis_connection().ping())
    except Exception:
        return False


def get_queue() -> Queue:
    return Queue(settings.rq_queue_name, connection=get_redis_connection())
