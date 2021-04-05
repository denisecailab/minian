import ruamel.yaml

"""
A simple script to convert environment.yml to requirements.txt
Requires:

conda install -c conda-forge ruamel.yaml

ref: https://gist.github.com/pemagrg1/f959c19ec18fee3ce2ff9b3b86b67c16
"""

yaml = ruamel.yaml.YAML()
data = yaml.load(open("../environment.yml"))

requirements = []
for dep in data["dependencies"]:
    if isinstance(dep, str):
        try:
            package, package_version, python_version = dep.split("=")
        except ValueError:
            package, package_version = dep.split("=")
            python_version = "None"
        if python_version == "0":
            continue
        requirements.append(package + "==" + package_version)
    elif isinstance(dep, dict):
        for preq in dep.get("pip", []):
            requirements.append(preq)

with open("../requirements/requirements-base.txt", "w") as fp:
    for requirement in requirements:
        print(requirement, file=fp)
