import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow import keras
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

EXCEL_PATH = "./Exam1-v7/excel/train_test.xlsx"
IMAGES_PATH = "./Exam1-v7/Data/"
OUTPUT_PATH = '/home/ubuntu/DL-Exam-1/DAY4_4_promising/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

BATCH_SIZE = 100
IMG_H = 64
IMG_W = 64
N_CLASSES = 10
EPOCHS = 200

INPUT_SHAPE = IMG_H*IMG_W*3

data_df = pd.read_excel(EXCEL_PATH)
print(data_df.head())
print("Number of classes : ", data_df["target"].nunique())

train_df = data_df[data_df['split'] == 'train']
test_df = data_df[data_df['split'] == 'test']

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)


train_paths = [IMAGES_PATH+img_id for img_id in train_df['id'].values]
test_paths = [IMAGES_PATH+img_id for img_id in test_df['id'].values]

import ast
train_labels = train_df['target_class'].apply(ast.literal_eval).tolist()
test_labels = test_df['target_class'].apply(ast.literal_eval).tolist()

import tensorflow as tf
from tensorflow.keras import layers


def random_sharpness(image, alpha_range=(0.5, 2.0)):
    """Applies random sharpness adjustment to an RGB image using a depthwise convolution."""
    alpha = tf.random.uniform([], alpha_range[0], alpha_range[1])  # Random sharpness factor

    # Define a sharpening kernel for RGB images (3 channels)
    kernel = tf.constant([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=tf.float32)  # 3x3 sharpening kernel
    kernel = tf.reshape(kernel, [3, 3, 1, 1])  # Reshape for TensorFlow conv2d
    kernel = tf.tile(kernel, [1, 1, 3, 1])  # Duplicate kernel across RGB channels
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    image = tf.squeeze(image, axis=0)  # Remove batch dimension
    return tf.clip_by_value(alpha * image + (1 - alpha) * image, 0.0, 1.0)  # Ensure values remain in [0,1]


# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomFlip("vertical"),
#     layers.RandomRotation(0.3),  # Increased from 0.2
#     layers.RandomZoom(0.2),  # Increased from 0.1
#     layers.RandomContrast(0.2),  # Increased from 0.1
#     layers.RandomBrightness(0.2),  # Add brightness augmentation
#     layers.Lambda(lambda img: random_sharpness(img, alpha_range=(0.5, 2.0)))  # Apply sharpness
#
# ])


# Augmentation function for training data

# Define the base augmentation pipeline
base_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    # layers.Lambda(lambda img: random_sharpness(img, alpha_range=(0.5, 2.0)))
])

# Define weaker augmentation (subset of base augmentation)
weak_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomBrightness(0.1),
])

# Define stronger augmentation (more transformations applied)
strong_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.4),  # Increased rotation
    layers.RandomZoom(0.3),  # Increased zoom
    layers.RandomContrast(0.3),  # Increased contrast
    layers.RandomBrightness(0.3),  # Increased brightness
    # layers.Lambda(lambda img: random_sharpness(img, alpha_range=(0.3, 2.5)))  # More sharpness variation
])


def augment_image(image, label):

    more_aug_classes = [9, 8, 2, 4]  # Strong augmentation
    less_aug_classes = [6, 3, 1, 5, 7]  # Weaker augmentation
    more_aug_condition = tf.reduce_any(tf.cast(tf.gather(label, more_aug_classes), tf.bool))
    less_aug_condition = tf.reduce_any(tf.cast(tf.gather(label, less_aug_classes), tf.bool))
    image = tf.cond(
        more_aug_condition,
        lambda: strong_augmentation(image, training=True),  # Apply strong augmentation
        lambda: tf.cond(
            less_aug_condition,
            lambda: weak_augmentation(image, training=True),  # Apply weak augmentation
            lambda: image  # No augmentation for well-represented classes
        )
    )

    # image = tf.reshape(image, [-1])
    # print(label)
    return image, label


# def load_image(image_path, label):
#     # print(image_path,label)
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMG_H, IMG_W])
#     image = tf.image.rgb_to_grayscale(image)
#     image = image / 255.0  # Normalize to [0, 1]
#     # tf.print("Shape after load_image:", tf.shape(image))  # Print shape after loading
#     return image, label

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0  # Normalize to [0, 1]

    # Reshape to remove channel dimension (IMG_H, IMG_W, 1) → (IMG_H, IMG_W)
    image = tf.reshape(image, [IMG_H, IMG_W])

    # Split into 72 separate vectors along the height axis
    image_slices = tf.split(image, IMG_H, axis=0)

    # Remove the extra dimension to make shape (IMG_W,) instead of (1, IMG_W)
    image_slices = [tf.squeeze(slice_, axis=0) for slice_ in image_slices]

    return tuple(image_slices), label  # Return as tuple of 72 vectors

# def load_image_test(image_path, label):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMG_H, IMG_W])
#     image = tf.image.rgb_to_grayscale(image)
#     image = image / 255.0  # Normalize to [0, 1]
#     image = tf.reshape(image, [-1])  # Flatten the test image into a 1D vector
#     return image, label

def load_image_test(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0  # Normalize to [0, 1]

    # Reshape to remove channel dimension (IMG_H, IMG_W, 1) → (IMG_H, IMG_W)
    image = tf.reshape(image, [IMG_H, IMG_W])

    # Split into 72 separate vectors along the height axis
    image_slices = tf.split(image, IMG_H, axis=0)

    # Remove the extra dimension to make shape (IMG_W,) instead of (1, IMG_W)
    image_slices = [tf.squeeze(slice_, axis=0) for slice_ in image_slices]

    return tuple(image_slices), label  # Return as tuple of 72 vectors

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
print(f"Size of training dataset: {len(train_ds)}")

# Apply transformations
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
# train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
print(f"Size of training dataset: {len(train_ds)}")
print(f"Size of test dataset: {len(test_ds)}")

# Shuffle, batch and prefetch
train_ds = train_ds.shuffle(len(train_ds)//12).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# train_ds = train_ds.concatenate(test_ds)  #<------- might drop this line later.

for image_batch, label_batch in train_ds.take(1):  # Only print for the first batch
    tf.print("Shape of image_batch after all transformations:", tf.shape(image_batch))

for image_batch, label_batch in test_ds.take(1):  # Only print for the first batch
    tf.print("Shape of test image_batch after all transformations:", tf.shape(image_batch))


# model = models.Sequential([
#     layers.Input(shape=(INPUT_SHAPE,)),
#     layers.Dense(8000),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Dropout(0.1),
#
#     # layers.Dense(7000),
#     # layers.BatchNormalization(),
#     # layers.ReLU(),
#     # layers.Dropout(0.3),
#     #
#     layers.Dense(3000),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Dropout(0.1),
#     # #
#     # layers.Dense(250),
#     # layers.BatchNormalization(),
#     # layers.ReLU(),
#     # layers.Dropout(0.3),
#     # #
#     layers.Dense(500),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Dropout(0.1),
#
#     layers.Dense(125),
#     layers.BatchNormalization(),
#     layers.ReLU(),
#     layers.Dropout(0.1),
#
#     layers.Dense(N_CLASSES, activation='softmax'),
# ])
# model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

# from tensorflow.keras import layers, Model
#
# def build_model(input_shape=INPUT_SHAPE, num_classes=N_CLASSES):
#     inputs = layers.Input(shape=(input_shape,))
#     x = layers.Dense(input_shape, activation='relu')(inputs)
#     x = layers.BatchNormalization()(x)
#     # x = layers.Dropout(0.3)(x)
#     x = layers.Add()([x, inputs])
#     x = layers.Dense(int(input_shape * 2.0 / 3), activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     # x = layers.Dropout(0.3)(x)
#
#     x = layers.Dense(750, activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     # x = layers.Dropout(0.3)(x)
#
#     x = layers.Dense(500, activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     # x = layers.Dropout(0.3)(x)
#
#     x = layers.Dense(125, activation='relu')(x)
#     # x = layers.BatchNormalization()(x)
#     # x = layers.Dropout(0.3)(x)
#
#     outputs = layers.Dense(num_classes, activation='softmax')(x)
#
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
# model = build_model()


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, BatchNormalization
from tensorflow.keras.models import Model


def build_parallel_mlp(output_dimension):
    num_vectors = IMG_H  # Number of parallel input vectors (72)
    vector_size = IMG_W  # Size of each input vector (72)

    inputs = [Input(shape=(vector_size,)) for _ in range(num_vectors)]

    processed_vectors = []
    for inp in inputs:
        x = Dense(52, activation='relu')(inp)
        x = Dense(24, activation='relu')(x)
        processed_vectors.append(x)

    merged = Concatenate()(processed_vectors)

    # Apply Batch Normalization only to merged output
    # x = BatchNormalization()(merged)
    x = Dense(125, activation='relu')(merged)

    # No Batch Norm before softmax
    outputs = Dense(output_dimension, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
# Example usage
output_dimension = 10  # Example: 10 output classes
model = build_parallel_mlp(output_dimension)
print(model.summary())

# try:
#     best_model_path = OUTPUT_PATH + "model_epoch_.keras"
#     model = load_model(best_model_path)
#     print("-----Model Loaded-----")
#     start_epoch = 51
# except:
#     print("-----Model Not Loaded-----")
#
#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=0.1,
#         decay_steps=939,
#         decay_rate=0.95,
#         staircase=False
#     )
#     optimizer = tf.keras.optimizers.Adam()
#     model.compile(
#         optimizer=optimizer,
#         loss="categorical_crossentropy",
#         metrics=['accuracy'])
#     start_epoch = 0

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("USING GPU")
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True
)


checkpoint_callback = ModelCheckpoint(
    OUTPUT_PATH+'model_epoch_{epoch:02d}.keras',
    monitor='val_accuracy',
    save_weights_only=False,
    save_best_only=True,
    verbose=1
)

class_weight_dict = {0: 0.44161370804500305, 6: 0.7705930671923772, 3: 0.9078873512048776,
                     1: 0.9155963302752294, 7: 1.0121048656813032, 5: 1.0321487512377598, 9: 1.3701182999853951,
                     8: 1.574290988420876, 2: 2.0500874125874127, 4: 2.3015701668302255}

history = model.fit(
    train_ds,
    initial_epoch = 0,
    epochs=EPOCHS,
    validation_data=test_ds,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_callback, early_stopping]
)

model.save(OUTPUT_PATH+'model_Alicia.keras')
# model.save('./model_Alicia.keras')

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


import os
import matplotlib.pyplot as plt

# Ensure the 'plots' directory exists
plots_dir = os.path.join(OUTPUT_PATH, "plots")
os.makedirs(plots_dir, exist_ok=True)

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plots_dir, "loss_plot.png"))
plt.close()

# Training and Validation Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(plots_dir, "accuracy_plot.png"))
plt.close()







