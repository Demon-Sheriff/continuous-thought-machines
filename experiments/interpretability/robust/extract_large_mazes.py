"""
Extract large maze data from tar.gz on Modal volume.
Usage: modal run experiments/interpretability/robust/extract_large_mazes.py
"""

import modal

app = modal.App("extract-large-mazes")
data_volume = modal.Volume.from_name("ctm-data", create_if_missing=True)


@app.function(
    timeout=600,
    volumes={"/data": data_volume},
)
def extract_large_mazes():
    """Extract the large.tar.gz file on the Modal volume."""
    import os
    import tarfile

    tar_path = "/data/mazes/large.tar.gz"
    extract_path = "/data/mazes"

    if not os.path.exists(tar_path):
        print(f"ERROR: {tar_path} not found!")
        return {"status": "error", "message": "tar.gz not found"}

    print(f"Extracting {tar_path} to {extract_path}...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

    # Commit changes to volume
    data_volume.commit()

    # Verify extraction
    large_path = "/data/mazes/large"
    if os.path.exists(large_path):
        contents = os.listdir(large_path)
        print(f"\nExtracted successfully! Contents of {large_path}:")
        for item in contents:
            item_path = os.path.join(large_path, item)
            if os.path.isdir(item_path):
                sub_contents = os.listdir(item_path)
                print(f"  {item}/ ({len(sub_contents)} items)")
            else:
                print(f"  {item}")

        # Clean up tar file
        os.remove(tar_path)
        data_volume.commit()
        print(f"\nRemoved {tar_path}")

        return {"status": "success", "contents": contents}
    else:
        return {"status": "error", "message": "Extraction failed"}


@app.local_entrypoint()
def main():
    result = extract_large_mazes.remote()
    print(f"\nResult: {result}")
