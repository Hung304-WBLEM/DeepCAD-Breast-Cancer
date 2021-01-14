import config.cfg_files as cfg
import os

from utils.fileio import json

proj_paths_json = json.read(os.path.join(cfg.CONFIG_ROOT, cfg.PROJECT_PATHS))
