#%%
''' 
Author: Owen A. Johnson
Date of Last Major Update: 01/07/2024
Code Purpose: Generating .gif files from images. 
'''

import re 
from glob import glob
import contextlib
from PIL import Image, ImageSequence
from tqdm import tqdm

def extract_number(filename):
    match = re.search(r'(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return 

def save_optimized_gif(input_filenames, output_filename, duration=4, loop=0):
    if f'{output_filename}' in glob('gifs/*'):
        print(f"Skipping {output_filename} as it already exists.")
        return
    try:
        images = []
        for filename in sorted(input_filenames):
            img = Image.open(filename)
            images.append(img.copy())
            img.close()
        
        if images:
            images[0].save(output_filename, save_all=True, append_images=images[1:], duration=duration, loop=loop, optimize=True)
        print(f"GIF saved successfully as {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

folder_list = glob('2*') # matches PDS EID folders

for folder in tqdm(folder_list):
    filenames = glob(f'{folder}/*.png')
    
    save_optimized_gif(filenames, f'gifs/{folder}.gif', duration=4, loop=0)