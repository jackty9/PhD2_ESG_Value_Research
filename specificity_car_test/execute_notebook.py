from pathlib import Path
import os
import sys
root = Path(__file__).resolve().parents[1]
os.environ['JUPYTER_PATH'] = str(root / '.runtime' / 'jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.runtime' / 'jupyter-runtime')
os.environ['IPYTHONDIR'] = str(root / '.runtime' / 'ipython')
import nbformat
from nbclient import NotebookClient
path = root / (sys.argv[1] if len(sys.argv) > 1 else 'CEO_Letter_Specificity_CAR_Test.ipynb')
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=240, kernel_name='specificity-car', resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(nb, path)
print('Notebook executed successfully; all output cells saved.')
