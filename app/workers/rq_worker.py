from rq import Worker

from app.config.settings import settings
from app.queue import get_redis_connection


def run_worker() -> None:
    redis_conn = get_redis_connection()
    worker = Worker([settings.rq_queue_name], connection=redis_conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    run_worker()
