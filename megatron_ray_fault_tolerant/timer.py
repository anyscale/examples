import time
from loguru import logger

class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        logger.opt(depth=1).info(f"Started: '{self.message}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"Finished: '{self.message}', time cost: {time.time() - self.start_time:.2f}s")

    async def __aenter__(self):
        self.start_time = time.time()
        logger.opt(depth=1).info(f"Started: '{self.message}'")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"Finished: '{self.message}', time cost: {time.time() - self.start_time:.2f}s")