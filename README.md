# PHD CVA Paper

Contains code for the CVA research line of my thesis.


# Instructions to manage python installations and environments with pyenv and uv:


🚀 Initialize a New Project (pyenv + uv)
## Create project
mkdir my_project

cd my_project

## Select Python
pyenv install 3.12.2

pyenv local 3.12.2

## Initialize project
uv init

## Add dependencies
uv add numpy

uv add jupyterlab ipykernel

## Create / sync environment
uv sync


## Files created:
pyproject.toml

uv.lock

.python-version

.venv/ (not committed)

Other should be committed.

📥 Clone an Existing Project (Reproduce Environment)
## Clone
git clone <REPO_URL>

cd <REPO_NAME>

## Install project Python
pyenv install -s $(cat .python-version)

pyenv local $(cat .python-version)

uv venv --python $(cat .python-version)

## Recreate environment
uv sync --frozen

## To run notebook or scripts from the shell
source .venv/bin/activate

.venv/Scripts/activate if windows


🧠 Mental Model (remember this)
pyenv  → Python version
uv     → deps + venv
uv add → declare deps
uv sync→ make env match project
s
