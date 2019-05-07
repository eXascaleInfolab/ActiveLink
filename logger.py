import logging
import os
import sys
import time


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s %(levelname)s  %(message)s")

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    file_name = int(time.time())

    file_handler = logging.FileHandler("{0}/{1}.log".format(log_dir, file_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
