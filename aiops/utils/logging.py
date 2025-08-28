import logging
import sys


def get_logger(name: str) -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler(stream=sys.stdout)
		formatter = logging.Formatter(
			"%(asctime)s | %(levelname)s | %(name)s | %(message)s",
			datefmt="%H:%M:%S",
		)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		logger.setLevel(logging.INFO)
	logger.propagate = False
	return logger

