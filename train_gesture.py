import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Dynamically set dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, 'gesture', 'train')
test_path = os.path.join(BASE_DIR, 'gesture', 'test')

# Ensure directories exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Dynamically get classes from the training directory
labels = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
n_classes = len(labels)

# Data preprocessing with validation split from training data
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    validation_split=0.2
)

train_batches = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(64,64),
    class_mode='categorical',
    batch_size=10,
    shuffle=True,
    subset='training'
)

validation_batches = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(64,64),
    class_mode='categorical',
    batch_size=10,
    shuffle=True,
    subset='validation'
)

# Model architecture using Input layer
inputs = tf.keras.Input(shape=(64, 64, 3))
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid')(x)
x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(n_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(
    train_batches,
    epochs=10,
    validation_data=validation_batches
)

# Save trained model
model.save('best_model_dataflair.h5')

print("Model training complete!")