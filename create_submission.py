import os
import shutil
import tarfile
from datetime import datetime


def create_submission_package(project_dir, output_filename="submission.tar.gz"):
    """
    Create a clean submission package for Kaggle.

    Args:
        project_dir (str): Path to the project directory
        output_filename (str, optional): Name of the output tar.gz file
    """
    # Essential files and directories to include
    essential_files = [
        "agent.py",
        "main.py",
        "core",
        "lux",  # Include the entire lux directory
        "__pycache__",  # Include the __pycache__ directory
    ]

    # Create submissions directory if it doesn't exist
    submissions_dir = os.path.join(project_dir, "submissions")
    os.makedirs(submissions_dir, exist_ok=True)

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_folder = os.path.join(submissions_dir, timestamp)
    os.makedirs(submission_folder, exist_ok=True)

    # Create temp working directory
    temp_dir = os.path.join(project_dir, "submission_temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Copy essential files
        for item in essential_files:
            src_path = os.path.join(project_dir, item)
            dst_path = os.path.join(temp_dir, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Create tar.gz archive with timestamp
        output_filename_with_timestamp = (
            f"{os.path.splitext(output_filename)[0]}_{timestamp}.tar.gz"
        )
        output_path = os.path.join(submissions_dir, output_filename_with_timestamp)
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(temp_dir, arcname=".")

        with open(os.path.join(submission_folder, "metadata.txt"), "w") as f:
            f.write(f"Submission created: {datetime.now().isoformat()}\n")
            f.write(f"Package Name: {output_filename_with_timestamp}\n")

        print(f"Submission package created in: {submission_folder}")
        print(f"Archive file: {output_filename_with_timestamp}")

    except Exception as e:
        print(f"Error creating submission package: {e}")
        if os.path.exists(submission_folder):
            shutil.rmtree(submission_folder)
        raise

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Use the script's directory as the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    create_submission_package(project_dir)
