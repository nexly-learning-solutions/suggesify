import os
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def check_git_installed():
    """Check if git is installed on the system."""
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        logger.error("Git is not installed or not found in the system PATH.")
        sys.exit(1)

def clone_repo(repo_url, target_dir, retries=3, delay=5):
    """Clone a repository with retries."""
    for attempt in range(retries):
        try:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = target_dir / repo_name
            if repo_path.exists():
                logger.info(f"Repository {repo_name} already exists, skipping clone.")
                return

            logger.info(f"Cloning {repo_url}...")
            subprocess.run(["git", "clone", repo_url], check=True, cwd=target_dir)
            logger.info(f"Successfully cloned {repo_url}")
            return
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning {repo_url} (Attempt {attempt + 1}): {e}")
            time.sleep(delay)
    logger.error(f"Failed to clone {repo_url} after {retries} attempts.")

def clone_repositories(directory, repos, max_workers=4):
    """Clone a list of repositories into a specified directory using concurrency."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(clone_repo, repo, path) for repo in repos]
        for future in as_completed(futures):
            try:
                future.result()  # To re-raise any exceptions caught during cloning
            except Exception as e:
                logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Clone multiple git repositories into a specified directory.")
    parser.add_argument("--directory", type=str, default=os.getenv("CLONE_DIR", "3rdparty"), help="Directory to create and clone repositories into")
    parser.add_argument("--workers", type=int, default=int(os.getenv("CLONE_WORKERS", 4)), help="Number of concurrent cloning processes")
    args = parser.parse_args()

    check_git_installed()

    repositories = [
        "https://github.com/OpenMathLib/OpenBLAS.git",
        "https://github.com/pybind/pybind11.git",
        "https://github.com/Unidata/netcdf-cxx4.git",
        "https://github.com/NVIDIA/TensorRT.git",
        "https://github.com/microsoft/Microsoft-MPI.git",
        "https://github.com/python/cpython.git",
        "https://github.com/NVIDIA/nccl.git",
        "https://github.com/pytorch/pytorch.git",
        "https://github.com/NVIDIA/cudnn-frontend.git",
        "https://github.com/open-source-parsers/jsoncpp.git",
        "https://github.com/google/highway.git",
        "https://github.com/google/googletest.git",
        "https://github.com/NVIDIA/cutlass.git",
        "https://github.com/GerHobbelt/pthread-win32.git",
        "https://github.com/boostorg/boost.git",
        "https://github.com/google/glog.git",
        "https://github.com/NVIDIA/cccl.git",
    ]

    clone_repositories(args.directory, repositories, max_workers=args.workers)
    logger.info("All repositories have been processed.")