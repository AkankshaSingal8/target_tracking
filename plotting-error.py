import os
from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras


import numpy as np
from matplotlib.image import imread
from tqdm import tqdm
from PIL import Image

import seaborn as sns
import pandas as pd

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])


model = tf.keras.models.load_model('model_singlestepfalse_b64_seqlen1_1to4.h5')
root = "./dataset/1"
image_paths = os.listdir(root)
image_paths.sort()
print(image_paths)
file_ending = 'png'

csv_file_name = "dataset/1/data_out.csv" 
labels = np.genfromtxt(csv_file_name, delimiter=',', skip_header=1, dtype=np.float32)
#print("labels", labels)
predictions = []


for path in image_paths:
    if file_ending in path:
        img = Image.open(root + '/' + path)
        img = img.resize(IMAGE_SHAPE_CV)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=1)
        prediction = model.predict(img_array)
        print(prediction)
        print(path)
        

#predictions = model.predict(preprocessed_images)
# df_predictions = pd.DataFrame(predictions, columns=['Image Path'] + [f'Prediction {i+1}' for i in range(prediction.shape[1])])
# df_predictions.to_csv(csv_file_name, index=False)

# print(f"Predictions saved to {csv_file_name}")