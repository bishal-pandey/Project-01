import logging
import os
from datetime import datetime



LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),"Logs",LOG_FILE)
os.makedirs(log_path)
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)


logging.basicConfig(filename=LOG_FILE_PATH, format="[%(asctime)s] - %(lineno)s %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG)

if __name__ == "__main__":
    logging.info("hello world")
    logging.warning("hello world")
    logging.critical("hello world")