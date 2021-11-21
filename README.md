# predicators

## Installation
### pip
* This repository uses Python versions 3.8+.
* Run `pip install -r requirements.txt` to install dependencies.

### Poetry
Run the below commands from a folder that will contain the `predicators` folder after install.

```
git clone https://github.com/Learning-and-Intelligent-Systems/predicators.git
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
cd PGMax
poetry shell
poetry install
```

## Instructions for running code:
### pip
* (recommended) Make a new virtual env or conda env
* Make sure the parent of the repository is on your PYTHONPATH.
* Run, e.g., `python src/main.py --env cover --approach oracle --seed 0` to run the system.

### Poetry
* Activate your poetry shell from within the repository with `poetry shell`
* Run, e.g., `python src/main.py --env cover --approach oracle --seed 0` to run the system.

## Instructions for contributing:
* You can't push directly to master. Make a PR and merge that in.
* To merge a PR, you have to pass 3 checks, all defined in `.github/workflows/predicators.yml`.
* The unit testing check verifies that tests pass and that code is adequately covered. To run locally: `pytest -s tests/ --cov-config=.coveragerc --cov=src/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered`, which will print out the lines that are uncovered in every file. The "100" here means that all lines in every file must be covered. If that turns out to be too stringent, we can decrease it later.
* The static typing check uses Mypy to verify type annotations. To run locally: `mypy .`. If this doesn't work due to import errors, try `mypy -p predicators` from one directory up.
* The linter check runs pylint with the custom config file `.predicators_pylintrc` in the root of this repository. Feel free to edit this file as necessary. To run locally: `pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc`.

### Using pip
* In addition to the packages in requirements.txt, please pip install the following packages if you want to contribute to the repository: pytest-cov>=2.12.1 and pytest-pylint>=0.18.0. Also, install mypy from source: pip install -U git+git://github.com/python/mypy.git@9a10967fdaa2ac077383b9eccded42829479ef31. (Note: if this mypy issue gets resolved, we can install from head again.)

### Using poetry
* If your contribution introduces a new dependency, be sure to add it to Poetry ([instructions here](https://python-poetry.org/docs/cli/#add)). In brief, if you add a new dependency that users will require (e.g. a new deep learning package, or something that affects some functionality of the codebaase), run: `poetry add <package-name>`. If you add a new dependency that's really only for developers (e.g. new linter or formatter), run: `poetry add -D <package-name>`.