import json
import sys

def convert_nb(path, out_path):
    with open(path, 'r') as f:
        nb = json.load(f)
    with open(out_path, 'w') as f:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                f.write("".join(cell['source']) + "\n\n")

convert_nb('train_models.ipynb', 'temp_train.py')
convert_nb('reproduce_results.ipynb', 'temp_reproduce.py')
