#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Mars Team FDL 2024 modified this code to work with the dataset we have

# load the libraries
import tqdm

import pathlib
import itertools
import collections

import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import pandas as pd

import keras
from keras import layers

import os
from glob import glob

import input_videos
import train_model_utils
from helper_functions import *


# Define the dimensions of one frame in the set of frames created
# All the file inputs are for our Mars dataset, this is where
# we would specify the file locations
# -----------------------------------------------
# AS, OJ: checked using cv2 library
#file_path = "/home/arushi/Desktop/FDL_2024/videos_subset/samurai_video_norm_S0003.mp4"
#vid = cv2.VideoCapture(file_path)
#height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT) 
#width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

HEIGHT = 600
WIDTH = 800

n_frames   = 10
batch_size = 2
n_classes = 10

# -----------------------------------------------

train_labels  = pd.read_hdf('../train_set.hdf')
train_samples = list(train_labels['Sample ID'])

videos_dir   = '/home/arushi/Desktop/FDL_2024/videos_subset/'

# we are not at testing yet, keep it like this for now
test_labels  = pd.read_hdf('../test_set.hdf')
test_samples = list(test_labels['Sample ID'])

# -----------------------------------------------
# create the train and test directories
os.system('mkdir "train/"')
os.system('mkdir "test/"')
file_names = glob(videos_dir + '*.mp4')
train_label_array = []
test_label_array  = []

for i in range (len(file_names)):
    sample_id = (file_names[i].split('/')[-1][-9:-4]) # get the sample name from the file

    # take care of mars samples that are strings
    if (sample_id[0] == '2'):
        sample_id = int(sample_id)

    if (sample_id in train_samples):
        os.system('cp ' + file_names[i] +  ' ' + ' train/')
        idx        = np.argwhere(str(sample_id) == np.array(train_samples))[0][0]
        label_dict = train_labels.iloc[idx]['Labels']
        train_label_array.append([int(value) for key, value in sorted(label_dict.items())])
        
    else:
        os.system('cp ' + file_names[i] + ' ' + ' test/')
        idx        = np.argwhere(str(sample_id) == np.array(test_samples))[0][0]
        label_dict = test_labels.iloc[idx]['Labels']
        test_label_array.append([int(value) for key, value in sorted(label_dict.items())])

# -----------------------------------------------

# This is where we would input the number of frames we want etc.
# and batch sizes.

# the output shape is the 10 labels identified
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (10,), dtype = tf.int16))

train_files = pathlib.Path('./train')
test_files  = pathlib.Path('./test') 

train_ds = tf.data.Dataset.from_generator(input_videos.FrameGenerator(train_files, n_frames, train_label_array, training=True),
                                          output_signature = output_signature)

# Batch the data
train_ds = train_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(input_videos.FrameGenerator(test_files, n_frames, test_label_array),
                                         output_signature = output_signature)

test_ds = test_ds.batch(batch_size)
frames, label = next(iter(train_ds))



#-----------------------------------------------
# This is where the main model is defined

input_shape = (None, n_frames, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = train_model_utils.Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = train_model_utils.ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Block 1
x = train_model_utils.add_residual_block(x, 16, (3, 3, 3))
x = train_model_utils.ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = train_model_utils.add_residual_block(x, 32, (3, 3, 3))
x = train_model_utils.ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = train_model_utils.add_residual_block(x, 64, (3, 3, 3))
x = train_model_utils.ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = train_model_utils.add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(n_classes, activation="sigmoid")(x)
model = keras.Model(input, x)

frames, label = next(iter(train_ds))
model.build(frames)


#-----------------------------------------------
# This is where the model is trained

n_epochs = 2

model.compile(loss = keras.losses.CategoricalCrossentropy(), 
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001), 
              metrics = ['accuracy'])

history = model.fit(x = train_ds,
                    epochs = n_epochs, 
                    validation_data = test_ds)

# plot_history(history)

def get_actual_predicted_labels(dataset): 
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)

  return actual, predicted


actual, predicted = get_actual_predicted_labels(train_ds)
idx = np.random.randint(0, len(train_label_array))
# --------------------------------------------------
# This is for plotting only
# plt.plot(predicted[idx], 'o-', label='predicted')
# plt.plot(actual[idx], 'o-', label='actual')
# plt.legend()
# plt.show()

### TODO: Add here about precision and recall
### TODO: clean up so that we can save the model run results
#### TODO: Think about how we can input the graph as one value since it is in grayscale
# this would mean we don't need [R, G, B] or 3 channels
