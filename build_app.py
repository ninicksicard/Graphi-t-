import os
import sys

python_exe = sys.executable
os.system(f"{python_exe} setup.py bdist_mac")
