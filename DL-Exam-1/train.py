import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

EXCEL_PATH = "./Exam1-v7/excel/train_test.xlsx"
IMAGES_PATH = "./Exam1-v7/Data/"
OUTPUT_PATH = '/home/ubuntu/DL-Exam-1/DAY2/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

BATCH_SIZE = 120
IMG_H = 64
IMG_W = 64
N_CLASSES = 10

INPUT_SHAPE = IMG_H*IMG_W*3

data_df = pd.read_excel(EXCEL_PATH)
print(data_df.head())
print("Number of classes : ", data_df["target"].nunique())

train_df = data_df[data_df['split'] == 'train']
test_df = data_df[data_df['split'] == 'test']

train_paths = [IMAGES_PATH+img_id for img_id in train_df['id'].values]
test_paths = [IMAGES_PATH+img_id for img_id in test_df['id'].values]

train_labels = train_df['target_class'].apply(eval).tolist()
test_labels = test_df['target_class'].apply(eval).tolist()


def load_image(path : str, label: list) -> tuple[np.ndarray, tf.Tensor]:

    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_H, IMG_W))
    image = image/255.0
    return image, tf.convert_to_tensor(label)

def load_image_test(path : str, label: list) -> tuple[np.ndarray, tf.Tensor]:

    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_H, IMG_W))
    image = image/255.0
    image = tf.reshape(image, [-1])
    return image, tf.convert_to_tensor(label)

from tensorflow.keras import layers, models


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# Function to apply augmentation
def augment_image(image, label):
    image = data_augmentation(image, training=True)  # Apply data augmentation
    image = tf.reshape(image, [-1])  # Flatten the image into a 1D vector
    return image, label

# Create Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

# Load images (Assuming `load_image` already normalizes and resizes)
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

# **Apply Augmentation ONLY to Training Data**
train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch, Shuffle, and Prefetch
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(INPUT_SHAPE,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(N_CLASSES, activation='softmax'),
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("USING GPU")
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(
    monitor='val_loss',    # Monitor the validation loss
    patience=5,            # Stop training after 5 epochs of no improvement
    restore_best_weights=True  # Restore the best weights after training stops
)

# Define ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    OUTPUT_PATH+'model_epoch_{epoch:02d}.keras',  # Save model after each epoch with epoch number in filename
    save_weights_only=False,       # Save the entire model (not just weights)
    save_best_only=False,          # Save model after every epoch (not only the best one)
    verbose=1                      # Display a message when the model is saved
)

model.fit(
    train_ds,
    epochs=50,
    validation_data=test_ds,
    callbacks=[early_stopping, early_stopping]
)

model.save(OUTPUT_PATH+'model_Alicia.keras')
model.save('./model_Alicia.keras')

with open(OUTPUT_PATH+'summary_Alicia.txt','w') as f:
    model.summary(print_fn=lambda x: f.write(x+"\n"))


predictions = model.predict(test_ds)
predicted_labels = np.argmax(predictions, axis=1)

results_df = pd.DataFrame({
    "id": test_df["id"].values,
    "target": test_df["target"].values,
    "split": "test",
    "results": [f"class{p+1}" for p in predicted_labels]
})

results_df.to_excel(OUTPUT_PATH+"results_Alicia.xlsx", index=False)





