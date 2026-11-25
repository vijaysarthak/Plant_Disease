"""
config.py — ALL folder paths and settings of the app, in ONE place.

Every other file imports what it needs from here, for example:
    from config import MODEL_FILE
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. Folders and files
# ----------------------------------------------------------------------------
# PROJECT_DIR = the folder that contains this config.py file
PROJECT_DIR = Path(__file__).resolve().parent

# Put the files from trained_model.zip (made by the Colab notebook) into this folder:
SAVED_MODEL_DIR = PROJECT_DIR / "saved_model"
MODEL_FILE = SAVED_MODEL_DIR / "plant_model.onnx"        # the trained network
MODEL_INFO_FILE = SAVED_MODEL_DIR / "model_info.json"    # class names, thresholds, temperature ...

ADVICE_FILE = PROJECT_DIR / "data" / "advice.csv"        # what to tell the farmer for each group
DATABASE_FILE = PROJECT_DIR / "database" / "predictions.db"   # SQLite database (created automatically)

# ----------------------------------------------------------------------------
# 2. How a photo is prepared.
#    MUST be the same as the "image settings" in the Colab training notebook (Step 2).
# ----------------------------------------------------------------------------
RESIZE_SIZE = 256     # first make the shorter side 256 pixels ...
IMAGE_SIZE = 224      # ... then cut out the 224 x 224 centre
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # average colour of ImageNet photos (red, green, blue)
IMAGENET_STD = [0.229, 0.224, 0.225]    # spread of the colours of ImageNet photos
