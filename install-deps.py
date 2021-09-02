#!/usr/bin/env python3
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "matplotlib",
                       "numpy",
                       "numba",
                       "scipy",
                       "scikit-image",
                       "opencv-python"])
