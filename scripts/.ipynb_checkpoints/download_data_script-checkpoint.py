import os
import subprocess
import sys

def download_dataset_func():
    # Create directories
    os.makedirs("data/iwildcam_v2.0/", exist_ok=True)
    os.makedirs("ckpts/", exist_ok=True)

    # Check if gdown is installed, if not install it
    try:
        import gdown
    except ImportError:
        print("gdown not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "gdown"])
        import gdown

    # Download dataset_subtree.csv
    print("Downloading dataset_subtree.csv...")
    gdown.download(id="1l3o4TL0Acq_xmmIUEJRejrxZEmQthG-U", output="data/iwildcam_v2.0/dataset_subtree.csv", quiet=False)

    # Download pretrained model
    print("Downloading species_class_model.pt...")
    gdown.download(id="19cMXfFew4c9RzwG8iU2GFPQdC6xbryoQ", output="ckpts/species_class_model.pt", quiet=False)

    print("Download completed.")

