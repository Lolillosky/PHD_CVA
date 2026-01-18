# PHD_CVA
Contains code and notebooks for my PHD research

pipenv install
pipenv run python -m ipykernel install --user --name PHD_CVA --display-name "Python 3.14 (PHD_CVA)"


Project Setup (Pipenv + VS Code + Jupyter)
This project uses Pipenv for dependency management and Jupyter notebooks inside VS Code.
Follow these steps to set up the project on a new machine.
1. Prerequisites
Python
Install Python 3.14.x (same version used to create the environment).
Verify:
python3.14 --version
⚠️ If Python 3.14 is not installed, install it first before continuing.
Pipenv
Install Pipenv (user-level install recommended):
python3.14 -m pip install --user pipenv
Verify:
pipenv --version
2. Clone the repository
git clone <REPOSITORY_URL>
cd <PROJECT_FOLDER>
3. Create the virtual environment (inside the project)
We keep the virtual environment inside the project folder so VS Code can auto-detect it.
macOS / Linux
export PIPENV_VENV_IN_PROJECT=1
Windows (PowerShell)
$env:PIPENV_VENV_IN_PROJECT=1
4. Install dependencies (reproducible)
pipenv install --dev
This installs exact versions from Pipfile.lock.
5. Open the project in VS Code
code .
Make sure you have the Python and Jupyter extensions installed in VS Code.
6. Select the Python interpreter (once)
In VS Code:
Cmd/Ctrl + Shift + P
Python: Select Interpreter
Choose:
.venv/bin/python
You should see in the status bar:
Python 3.14.x (.venv)
7. Register the Jupyter kernel (required for notebooks)
Run this once per machine:
pipenv run python -m ipykernel install \
  --user \
  --name phd_cva \
  --display-name "PHD_CVA · Python 3.14"
8. Select the kernel in notebooks
Open any .ipynb file
Click Select Kernel (top right)
Choose:
PHD_CVA · Python 3.14
VS Code will remember this choice.
9. Verify everything works
In a notebook cell:
import sys
sys.executable
Expected output:
.../PROJECT_FOLDER/.venv/bin/python
Test key dependencies:
import numpy, scipy, pandas, torch, tensorboard
print("Environment OK")
10. Running TensorBoard
From the project root:
pipenv run tensorboard --logdir runs
Open the URL shown in the terminal (usually http://localhost:6006).
Notes
✅ Pipfile and Pipfile.lock are committed to Git
❌ .venv/ is not committed
❌ Jupyter kernels are not committed (local to each machine)
Troubleshooting
If VS Code asks again for an interpreter or kernel:
Re-select .venv/bin/python
Re-select the kernel PHD_CVA · Python 3.14
Reload VS Code window (Cmd/Ctrl + Shift + P → Reload Window)
Summary
git clone <repo>
cd <project>
export PIPENV_VENV_IN_PROJECT=1
pipenv install --dev
code .
pipenv run python -m ipykernel install --user --name phd_cva --display-name "PHD_CVA · Python 3.14"


