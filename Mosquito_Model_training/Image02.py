import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:/git_workspace/AIML_TeamWork/Mosquito_Model_training'):
    for filename in filenames:
        os.path.join(dirname, filename)

from tensorflow.data import AUTOTUNE
import tensorflow as tf
import matplotlib.pyplot as plt

Batch_SIZE = 128
NUM_STEPS = 64
img_height = 180
img_width = 180
data_dir = "zw4p9kj6nt-2/"

train_ds = tf.keras.utils.image_dataset_from_directory(
  "zw4p9kj6nt-2/data_splitting/Train/",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=Batch_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "zw4p9kj6nt-2/data_splitting/Test/",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=Batch_SIZE)  

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    tf.keras.layers.Conv2D(32, (3,3),1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.BatchNormalization(axis=3, momentum = 0.99, epsilon=0.001),
    tf.keras.layers.MaxPool2D((2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3),1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.BatchNormalization(axis=3, momentum = 0.99, epsilon=0.001),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

checkpoint_path = "C:/git_workspace/AIML_TeamWork/Mosquito_Model_training/training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath= checkpoint_path,
        verbose=1,
        save_weights_only=True
    )
]
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(
  train_ds,
  validation_data = val_ds,
  epochs=10,
  callbacks=callbacks
)

model.summary()

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')
plt.show()
print("")

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.show()