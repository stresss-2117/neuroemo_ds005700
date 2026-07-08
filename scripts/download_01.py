"""
Script 1: Download NeuroEmo dataset (ds005700)
"""

import openneuro
import os


def download_metadata():
    """Download dataset-level metadata files."""
    print("Downloading metadata...")
    try:
        openneuro.download(
            dataset="ds005700",
            target_dir="data/raw",
            include=[
                "dataset_description.json",
                "README",
            ]
        )
        print("Metadata downloaded!")
    except Exception as e:
        print(f"Metadata download warning: {e}")


def download_subject(subject_num):
    """Download one subject's data."""
    sub_id = f"sub-{subject_num:02d}"
    print(f"\n{'='*50}")
    print(f"Downloading {sub_id}...")
    print(f"{'='*50}")

     # Check if already downloaded
    bold_rest = f"data/raw/{sub_id}/func/{sub_id}_task-rest_bold.nii.gz"
    bold_fe   = f"data/raw/{sub_id}/func/{sub_id}_task-fe_bold.nii.gz"

    if os.path.exists(bold_rest) and os.path.exists(bold_fe):
        print(f"  {sub_id} already downloaded! Skipping.")
        return True

    try:
        openneuro.download(
            dataset="ds005700",
            target_dir="data/raw",
            include=[
                # Resting state only
            f"{sub_id}/func/{sub_id}_task-rest_bold.nii.gz",
            f"{sub_id}/func/{sub_id}_task-rest_bold.json",
            # Emotion task
            f"{sub_id}/func/{sub_id}_task-fe_bold.nii.gz",
            f"{sub_id}/func/{sub_id}_task-fe_bold.json",
            ]
        )
        print(f"  {sub_id} downloaded successfully!")
        return True

    except Exception as e:
        print(f"  ERROR downloading {sub_id}: {e}")
        return False


if __name__ == "__main__":
    download_metadata()
    for i in range(1, 6):
        download_subject(i)