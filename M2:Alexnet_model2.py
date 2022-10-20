# Dl framwork - tensorflow, keras a backend
import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings(&#39;ignore&#39;)
model = Sequential()
# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(250,250,3), kernel_size=(11,11),
strides=(4,4), padding=&#39;valid&#39;))
model.add(Activation(&#39;relu&#39;))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=&#39;valid&#39;))
# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1),
padding=&#39;valid&#39;))
model.add(Activation(&#39;relu&#39;))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding=&#39;valid&#39;))
# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
padding=&#39;valid&#39;))
model.add(Activation(&#39;relu&#39;))
# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation(&#39;relu&#39;))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation(&#39;relu&#39;))
# Add Dropout
model.add(Dropout(0.4))
# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation(&#39;relu&#39;))
# Add Dropout
model.add(Dropout(0.4))
# Output Layer
model.add(Dense(5))
model.add(Activation(&#39;softmax&#39;))
model.summary()
# Compile the model
model.compile(loss = &#39;categorical_crossentropy&#39;, optimizer=&#39;adam&#39;,
metrics=[&#39;accuracy&#39;])
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=
0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory(&#39;data/train&#39;,target_size=(25
0,250),batch_size=32,class_mode=&#39;categorical&#39;)
test_set=test_datagen.flow_from_directory(&#39;data/test&#39;,target_size=(250,250)
,batch_size=32,class_mode=&#39;categorical&#39;)
img_dims = 150
epochs = 10
batch_size = 64
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#### Fitting the model
history = model.fit(
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
plt.show()s
graph()
