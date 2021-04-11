import argparse
import os
import requests

NOTEBOOK_FILES = [
    "pipeline.ipynb",
    "cross-registration.ipynb",
    "img/workflow.png",
    "img/param_pnr.png",
    "img/param_spatial_update.png",
    "img/param_temporal_update.png",
]
DEMO_FILES = [f"demo_movies/msCam{i}.avi" for i in range(1, 11)] + [
    f"demo_data/session{i}/minian.nc" for i in range(1, 3)
]


def _get_file(filename: str, branch: str):
    if os.path.isfile(f"{filename}"):
        print(f"File {filename} already exists, skipping install of this file.")
        return

    r = requests.get(f"https://raw.github.com/DeniseCaiLab/minian/{branch}/{filename}")
    if r.status_code == 200:
        with open(f"{filename}", "wb") as f:
            for chunk in r.iter_content(2048):
                f.write(chunk)
        print(f"File {filename} installed.")


def demo(branch: str):
    os.makedirs("demo_movies", exist_ok=True)
    os.makedirs("demo_data", exist_ok=True)
    print("Installing demo data")
    for file in DEMO_FILES:
        _get_file(file, branch)


def notebook(branch: str):
    os.makedirs("img", exist_ok=True)
    print("Installing notebooks")
    for file in NOTEBOOK_FILES:
        _get_file(file, branch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebooks", action="store_true", help="Installs the notebooks"
    )
    parser.add_argument("--demo", action="store_true", help="Installs the demo data")
    parser.add_argument(
        "-b",
        action="store",
        default="master",
        help="Git repo branch name, default master",
    )
    args = parser.parse_args()

    branch = args.b
    print(f"Using branch: {branch}")

    if args.notebooks:
        notebook(branch)

    if args.demo:
        demo(branch)
