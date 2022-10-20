from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,
EarlyStopping
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings(&#39;ignore&#39;)
Classifier=Sequential()
Classifier.add(Convolution2D(32,3,3,input_shape=(250,250,3),activation=&#39;rel
u&#39;))
Classifier.add(MaxPooling2D(pool_size=(2,2)))
Classifier.add(Convolution2D(128,3,3,activation=&#39;relu&#39;))
Classifier.add(MaxPooling2D(pool_size=(2,2)))
Classifier.add(Flatten())
Classifier.add(Dense(256, activation=&#39;relu&#39;))
Classifier.add(Dense(5, activation=&#39;softmax&#39;))
Classifier.compile(optimizer=&#39;rmsprop&#39;,loss=&#39;categorical_crossentropy&#39;,metr
ics=[&#39;accuracy&#39;])
Classifier.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=
0.2,horizontal_flip=&#39;True&#39;)
training_set=train_datagen.flow_from_directory(&#39;data/train&#39;,target_size=(25
0,250),batch_size=32,class_mode=&#39;categorical&#39;)
test_set =
train_datagen.flow_from_directory(&#39;data/test&#39;,target_size=(250,250),batch_s
ize=32,class_mode=&#39;categorical&#39;)

from IPython.display import display
img_dims=150
epochs=40
batch_size=32
history =Classifier.fit_generator( training_set,
steps_per_epoch=training_set.samples // batch_size,
epochs=epochs,
validation_data=test_set,validation_steps=test_set.samples //
batch_size)
import h5py
Classifier.save(&#39;object_.h5&#39;)
#tf.lite.TFLiteConverter.from_keras_model(&#39;object_&#39;)
from keras.models import load_model
model=load_model(&#39;object_.h5&#39;)
import numpy as np
from tensorflow.keras.preprocessing import image
test_image=image.load_img(&#39;car_3.jpg&#39;,target_size=(250,250))
import matplotlib.pyplot as plt
img = plt.imshow(test_image)
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
prediction = result[0]
classes=training_set.class_indices
classes
prediction=list(prediction)
prediction
classes=[&#39;bike&#39;,&#39;cars&#39;,&#39;flowers&#39;,&#39;horse&#39;,&#39;human&#39;]
output=zip(classes,prediction)
output=dict(output)
output
if output[&#39;bike&#39;]==1.0 :
print(&#39;bike&#39;)
elif output[&#39;cars&#39;]==1.0:
print(&#39;cars&#39;)
elif output[&#39;flowers&#39;]==1.0:
print(&#39;flowers&#39;)
elif output[&#39;horse&#39;]==1.0:
print(&#39;horse&#39;)
elif output[&#39;human&#39;]==1.0:
print(&#39;human&#39;)
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
