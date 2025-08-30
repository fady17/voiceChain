

#### **`voiceChain/utils/logging.py`**

import sys
from loguru import logger

def setup_logging(level="INFO"):
    """
    Configures the Loguru logger for the application.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/voiceChain_{time}.json",
        level="DEBUG",
        serialize=True,
        rotation="10 MB",
        catch=True
    )
    logger.info("Logger configured.")