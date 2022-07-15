import torch
from matplotlib import cm
import matplotlib.pyplot as plt
from alse.test_function.fxns import yf
import gpytorch
from PIL import Image
import glob

frames = []
imgs = sorted(glob.glob("combined_*.png"))
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('combined_result.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=600, loop=0)

frames = []
imgs = sorted(glob.glob("iter_*.png"))
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('posterior_result.gif', format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=600, loop=0)