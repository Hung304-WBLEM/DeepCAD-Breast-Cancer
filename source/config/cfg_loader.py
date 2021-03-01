import config.cfg_files as cfg
import os

from utilities.fileio import json

proj_paths_json = json.read(os.path.join(cfg.CONFIG_ROOT, cfg.PROJECT_PATHS))

if __name__ == '__main__':
    print()
    os.path.exists()
