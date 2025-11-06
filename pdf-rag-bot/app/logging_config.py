from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO", enqueue=True,
           format="<green>{time}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}")
