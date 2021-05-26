import logging
# logging.basicConfig(format='[%(asctime)s] [%(levelname)s]: %(message)s', level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')

formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# logging.warning("Logger level %s" %logging.getLogger('root'))

logger = logging.getLogger("ExCut")
logger. setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logging.warning("Logger level %s" % logger)

# logger=logger
