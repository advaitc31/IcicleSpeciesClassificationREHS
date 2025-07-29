import os
import subprocess
import urllib.request
import tarfile

def setup_ott_taxonomy():
    # Download ott3.3.tgz
    url = "http://files.opentreeoflife.org/ott/ott3.3/ott3.3.tgz"
    tgz_path = "ott3.3.tgz"

    if not os.path.exists(tgz_path):
        print("Downloading ott3.3.tgz...")
        urllib.request.urlretrieve(url, tgz_path)
        print("Download complete.")
    else:
        print("ott3.3.tgz already exists, skipping download.")

    # Extract ott3.3.tgz
    print("Extracting ott3.3.tgz...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete.")
