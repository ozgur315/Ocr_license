import logging
from utils import constant_parameters


def get_logger():
    """
        Define the logger
    """
    # Creating logger instance
    logger = logging.getLogger(__name__)

    # Get the log level from constant_paramaters.py
    # 'INFO' is the default value.
    # If the attribute LOG LEVEL doesn't exist in the dev module, it defaults to 'INFO'.
    log_level = getattr(constant_parameters, 'LOG_LEVEL', 'INFO')

    # use the log level to set the level of logging.
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create a console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, log_level.upper()))

    # Create the formatter for the log file.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Adding formatter to stream handler
    stream_handler.setFormatter(formatter)

    # Adding stream handler to logger
    logger.addHandler(stream_handler)

    return logger


# define the logger instance
log = get_logger()
