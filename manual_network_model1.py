import os
import numpy as np
import matplotlib.pyplot as plt
# Dl framwork - tensorflow, keras a backend
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout,
BatchNormalization
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D,
LeakyReLU, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,
EarlyStopping
from IPython.display import display
from os import listdir
from os.path import isfile, join
from PIL import Image
import glob
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings(&#39;ignore&#39;)
dir_name_train_bike = &#39;data/train/bike&#39;
dir_name_train_car = &#39;data/train/cars&#39;
dir_name_train_flower = &#39;data/train/flowers&#39;
dir_name_train_human = &#39;data/train/human&#39;
dir_name_train_horse = &#39;data/train/horse&#39;
def plot_images(item_dir, n=5):
all_item_dir = os.listdir(item_dir)
item_files = [os.path.join(item_dir, file) for file in
all_item_dir][:n]
plt.figure(figsize=(35, 10))
for idx, img_path in enumerate(item_files):
plt.subplot(2, n, idx+1)
img = plt.imread(img_path)
plt.imshow(img, cmap=&#39;gray&#39;)
plt.axis(&#39;off&#39;)
plt.tight_layout()
def Images_details_Print_data(data, path):
print(&quot; ====== Images in: &quot;, path)
for k, v in data.items():
print(&quot;%s:\t%s&quot; % (k, v))
def Images_details(path):

files = [f for f in glob.glob(path + &quot;**/*.*&quot;, recursive=True)]
data = {}
data[&#39;images_count&#39;] = len(files)
data[&#39;min_width&#39;] = 10**100 # No image will be bigger than that
data[&#39;max_width&#39;] = 0
data[&#39;min_height&#39;] = 10**100 # No image will be bigger than that
data[&#39;max_height&#39;] = 0
for f in files:
im = Image.open(f)
width, height = im.size
data[&#39;min_width&#39;] = min(width, data[&#39;min_width&#39;])
data[&#39;max_width&#39;] = max(width, data[&#39;max_height&#39;])
data[&#39;min_height&#39;] = min(height, data[&#39;min_height&#39;])
data[&#39;max_height&#39;] = max(height, data[&#39;max_height&#39;])
Images_details_Print_data(data, path)
print(&quot;&quot;)
print(&quot;Trainned data for BIKES :&quot;)
print(&quot;&quot;)
Images_details(dir_name_train_bike)
print(&quot;&quot;)
plot_images(dir_name_train_bike)
print(&quot;&quot;)
print(&quot;Trainned data for CARS:&quot;)
print(&quot;&quot;)
Images_details(dir_name_train_car)
print(&quot;&quot;)
plot_images(dir_name_train_car, 10)
print(&quot;&quot;)
print(&quot;Trainned data for FLOWERS:&quot;)
print(&quot;&quot;)
Images_details(dir_name_train_flower)
print(&quot;&quot;)
plot_images(dir_name_train_flower, 10)
print(&quot;&quot;)
print(&quot;Trainned data for HORSE:&quot;)
print(&quot;&quot;)
Images_details(dir_name_train_horse)
print(&quot;&quot;)
plot_images(dir_name_train_horse, 10)
print(&quot;&quot;)
print(&quot;Trainned data for HUMAN:&quot;)
print(&quot;&quot;)
Images_details(dir_name_train_human)
print(&quot;&quot;)
plot_images(dir_name_train_human, 10)
Classifier = Sequential()
Classifier.add(Convolution2D(32, (3,3), input_shape =
(100,100,3),activation = &quot;relu&quot;))
Classifier.add(MaxPool2D(pool_size = (2,2)))
Classifier.add(Flatten())
Classifier.add(Dense(38, activation =&#39;relu&#39;))
Classifier.add(Dense(5, activation=&#39;softmax&#39;))

Classifier.compile(optimizer=&#39;rmsprop&#39;,loss=&#39;categorical_crossentropy&#39;,metr
ics=[&#39;accuracy&#39;])
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=
0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory(&#39;data/train&#39;,target_size=(10
0,100),batch_size=32,class_mode=&#39;categorical&#39;)
test_set=test_datagen.flow_from_directory(&#39;data/train&#39;,target_size=(100,100
),batch_size=32,class_mode=&#39;categorical&#39;)
img_dims = 150
epochs = 5
batch_size = 32
#### Fitting the model
history = Classifier.fit_generator(
training_set, steps_per_epoch=training_set.samples //
batch_size,
epochs=epochs,
validation_data=test_set,validation_steps=test_set.samples //
batch_size)
def graph():
#Plot training &amp; validation accuracy values
plt.plot(history.history[&#39;accuracy&#39;])
plt.plot(history.history[&#39;val_accuracy&#39;])
plt.title(&#39;Model accuracy&#39;)
plt.ylabel(&#39;Accuracy&#39;)
plt.xlabel(&#39;Epoch&#39;)
plt.legend([&#39;Train&#39;, &#39;Test&#39;], loc=&#39;upper left&#39;)
plt.show()
# Plot training &amp; validation loss values
plt.plot(history.history[&#39;loss&#39;])
plt.plot(history.history[&#39;val_loss&#39;])
plt.title(&#39;Model loss&#39;)
plt.ylabel(&#39;Loss&#39;)
plt.xlabel(&#39;Epoch&#39;)
plt.legend([&#39;Train&#39;, &#39;Test&#39;], loc=&#39;upper left&#39;)
plt.show()
graph()
