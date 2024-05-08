import matplotlib.pyplot as plt
import PIL
from PIL import Image
import matplotlib.image as mpimg
import os
import numpy as np
import seaborn as sns
import warnings



tumor_images = 'data\\Tumor'
healthy_images = 'data\\Healthy'

tumor_data = [i for i in os.listdir(tumor_images)]
healthy_data = [i for i in os.listdir(healthy_images)]


