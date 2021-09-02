#!/usr/bin/env python3
import subprocess
import sys

for step in ["conditions.py", "flora.py", "fauna.py"]:
    subprocess.check_call([sys.executable, step])
