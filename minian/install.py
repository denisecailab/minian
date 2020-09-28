import os
import requests

PIPELINE_FILES = ('pipeline.ipynb','pipeline_noted.ipynb','cross-registration.ipynb')
DEMO_FILES = [f'demo_movies/msCam{i}.avi' for i in range(1,11)]

def _ask_branch() -> str:
  return input("use branch [master]: ") or "master"

def _get_file(filename: str, branch: str):
  if os.path.isfile(f'{filename}'):
    print(f'File {filename} already exists, skipping install of this file.')
    return

  r = requests.get(f'https://raw.github.com/DeniseCaiLab/minian/{branch}/{filename}')
  if r.status_code == 200:
    with open(f'{filename}','wb') as f:
      for chunk in r.iter_content(2048):
        f.write(chunk)
    print(f'File {filename} installed.')

def demo():
  try:
    os.mkdir('demo_movies')
    print('Installing demo movies')
    branch = _ask_branch()
    for file in DEMO_FILES:
      _get_file(file, branch)
  except OSError:
    print('Creation of the directory demo_movies failed, not installing.')

def pipeline():
  print('Installing pipeline notebooks')
  branch = _ask_branch()
  for file in PIPELINE_FILES:
    _get_file(file, branch)

