import argparse
import os
import requests

PIPELINE_FILES = ("pipeline.ipynb", "cross-registration.ipynb")
DEMO_FILES = [f"demo_movies/msCam{i}.avi" for i in range(1, 11)]


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
    try:
        os.mkdir("demo_movies")
        print("Installing demo movies")
        for file in DEMO_FILES:
            _get_file(file, branch)
    except OSError:
        print("Creation of the directory demo_movies failed, not installing.")


def pipeline(branch: str):
    print("Installing pipeline notebooks")
    for file in PIPELINE_FILES:
        _get_file(file, branch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebooks", action="store_true", help="Installs the pipeline notebooks"
    )
    parser.add_argument("--demo", action="store_true", help="Installs the demo movies")
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
        pipeline(branch)

    if args.demo:
        demo(branch)
