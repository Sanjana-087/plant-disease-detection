"""
Logging utility for the Plant Disease Detection System.

Use get_logger(__name__) in every module instead of bare print() statements
(except for intentional training progress output in train.py).
"""

import logging
import sys
from pathlib import Path

from training.config import PROJECT_ROOT, RESULTS_PATH

# Default log directory under results/
LOG_DIR = Path(RESULTS_PATH) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Standard format: timestamp, level, module, message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track loggers already configured to avoid duplicate handlers
_configured_loggers: set[str] = set()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_filename: str | None = None,
) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (e.g. logging.DEBUG, logging.INFO).
        log_to_file: If True, also write logs to results/logs/.
        log_filename: Optional log file name. Defaults to '{name}.log'
            with dots replaced by underscores.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler — stdout for INFO and below, stderr for WARNING+
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        safe_name = (log_filename or name).replace(".", "_").replace("/", "_")
        log_path = LOG_DIR / f"{safe_name}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured_loggers.add(name)
    return logger


def setup_root_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Configure a root-style project logger for scripts run from CLI.

    Args:
        level: Logging level for the project root logger.

    Returns:
        Logger named 'plant_disease_detection'.
    """
    return get_logger("plant_disease_detection", level=level, log_filename="project")


def log_system_info(logger: logging.Logger) -> None:
    """
    Log Python version, project root, and TensorFlow availability.

    Args:
        logger: Logger instance to write messages to.

    Returns:
        None
    """
    import platform

    logger.info("Platform: %s", platform.platform())
    logger.info("Python: %s", platform.python_version())
    logger.info("Project root: %s", PROJECT_ROOT)

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        logger.info("TensorFlow: %s", tf.__version__)
        logger.info("GPUs available: %s", len(gpus))
        for i, gpu in enumerate(gpus):
            logger.info("  GPU %d: %s", i, gpu.name)
    except ImportError:
        logger.warning("TensorFlow is not installed.")


# Module-level default logger for quick imports
logger = get_logger(__name__)
