import os
import shutil
import tarfile
import sys


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
        "base.py",
        "debug.py",
        "pathfinding.py",
        "lux",  # Include the entire lux directory
        "__pycache__",  # Include the __pycache__ directory
    ]

    # Create a temporary submission directory
    submission_dir = os.path.join(project_dir, "submission_temp")
    os.makedirs(submission_dir, exist_ok=True)

    try:
        # Copy essential files
        for item in essential_files:
            src_path = os.path.join(project_dir, item)
            dst_path = os.path.join(submission_dir, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Create tar.gz archive
        output_path = os.path.join(project_dir, output_filename)
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(submission_dir, arcname=".")

        print(f"Submission package created: {output_path}")

    except Exception as e:
        print(f"Error creating submission package: {e}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(submission_dir, ignore_errors=True)


if __name__ == "__main__":
    # Use the script's directory as the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    create_submission_package(project_dir)
