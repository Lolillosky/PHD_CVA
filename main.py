import torch
import os
import sys
import numpy as np

def main():
    print("Hello from phd-cva!")

    import sys, os, platform
    print("PY:", sys.executable)
    print("ARCH:", platform.architecture())
    print("VENV:", os.environ.get("VIRTUAL_ENV"))


if __name__ == "__main__":
    main()
