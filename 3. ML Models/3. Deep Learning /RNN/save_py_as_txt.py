"""
this helper function saves the entire config file to the corresponding model directory
so we can keep track of the model architecture for the run
"""
import os

def write_python_file(filename, target_dir):
    with open(filename) as f:
        data = f.read()
        f.close()

    with open(os.path.join(target_dir,"traninig_config.txt"), mode="w") as f:
        f.write(data)
        f.close()
