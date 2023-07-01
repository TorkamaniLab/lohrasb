__version__ = "3.5.0"

import logging
import logging.config
import os
from pathlib import Path, PurePath

import ray
import yaml
from dotenv import load_dotenv

from .project_conf import ROOT_PROJECT

PATH_TO_LOG_CONF = ROOT_PROJECT / "lohrasb" / "config.yaml"

# DEFAULT_LEVEL in production env
DEFAULT_LEVEL = logging.ERROR


def log_setup(log_cfg_path=PATH_TO_LOG_CONF):
    try:
        with open(PATH_TO_LOG_CONF, "r") as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
            # set up logging configuration
            return True
    except Exception as e:
        print(e)
        print(
            f"In this module, the default logging will be applied. The error is {e} which will be skipped!"
        )
        return False


if log_setup():
    # create logger
    load_dotenv()
    env = os.getenv("env")
    logger = logging.getLogger(env)

else:
    print("default logger setting is applied !")
    logging.basicConfig(level=DEFAULT_LEVEL)
    logger = logging.getLogger()


# # Use Ray to accelerate computing
# try:
#     # Try to connect to Ray cluster.
#     ray.init(address="auto", ignore_reinit_error=True)
#     logger.info("Connected to Ray cluster!")
# except Exception as e:
#     # If connection fails, start Ray locally.
#     ray.init()
#     logger.warning("This error happened {e}. So Ray Started locally.")
