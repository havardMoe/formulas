from pathlib import Path
import nbformat

paths = Path('.').glob('E?_*.ipynb')

notebooks = [nbformat.read(path, 4) for path in paths]

# Creating a new notebook
final_notebook = nbformat.v4.new_notebook(metadata=notebooks[0].metadata)
final_notebook.cells = notebooks[0].cells

for notebook in notebooks[1:]:
    # Concatenating the notebooks
    final_notebook.cells += notebook.cells

# Saving the new notebook 
nbformat.write(final_notebook, 'final_notebook.ipynb')

