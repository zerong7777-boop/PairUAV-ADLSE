# setup path to CroCo to import ViT blocks
import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
CROCO_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../croco'))
CROCO_MODELS_PATH = path.join(CROCO_REPO_PATH, 'models')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(CROCO_MODELS_PATH):
    # workaround for sibling import
    sys.path.insert(0, CROCO_REPO_PATH)
else:
    raise ImportError(f"croco is not initialized, could not find: {CROCO_MODELS_PATH}.\n "
                      "Ensure the vendored croco/ source directory is present in the repository root.")
