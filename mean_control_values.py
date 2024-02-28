import numpy as np
from matplotlib.image import imread
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

root = 'dataset'
for directory in range(1, 11):
    csv_file_name = root + "/" + str(directory) + '/data_out.csv'
    labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
    print("labels", labels)
    means = np.mean(labels, axis = 0)
    std_deviation = np.std(labels, axis = 0)
    print("means", means)
    print("std_deviation", std_deviation)