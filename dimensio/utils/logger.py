import logging
import sys
from typing import Optional


DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LEVEL = logging.INFO


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(
    level: Optional[int] = None,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
):
    level = level or DEFAULT_LEVEL
    format_string = format_string or DEFAULT_FORMAT
    
    # Configure root logger for the compressor package
    logger = logging.getLogger('compressor')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False


def disable_logging():
    logger = logging.getLogger('compressor')
    logger.disabled = True


def enable_logging():
    logger = logging.getLogger('compressor')
    logger.disabled = False


setup_logging()