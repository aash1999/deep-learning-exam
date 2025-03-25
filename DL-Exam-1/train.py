import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

EXCEL_PATH = "./Exam1-v7/excel/train_test.xlsx"
IMAGES_PATH = "./Exam1-v7/Data/"
OUTPUT_PATH = '/home/ubuntu/DL-Exam-1/DAY4/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

BATCH_SIZE = 64
IMG_H = 150
IMG_W = 150
N_CLASSES = 10
EPOCHS = 200

INPUT_SHAPE = IMG_H*IMG_W*3

data_df = pd.read_excel(EXCEL_PATH)
print(data_df.head())
print("Number of classes : ", data_df["target"].nunique())

train_df = data_df[data_df['split'] == 'train']
test_df = data_df[data_df['split'] == 'test']

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


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.3),  # Increased from 0.2
    layers.RandomZoom(0.2),  # Increased from 0.1
    layers.RandomContrast(0.2),  # Increased from 0.1
    layers.RandomBrightness(0.2),  # Add brightness augmentation
    layers.Lambda(lambda img: random_sharpness(img, alpha_range=(0.5, 2.0)))  # Apply sharpness

])


# Augmentation function for training data
def augment_image(image, label):
    # Define the underrepresented classes as indices
    underrepresented_classes = [2, 4, 8]
    label_condition = tf.reduce_any(tf.cast(tf.gather(label, underrepresented_classes), tf.bool))
    image = tf.cond(
        label_condition,  # If the condition is true, apply augmentation
        lambda: data_augmentation(image, training=True),
        lambda: image  # Otherwise, return the image unchanged
    )
    image = tf.reshape(image, [-1])
    return image, label

def load_image(image_path, label):
    print(image_path,label)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    # image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0  # Normalize to [0, 1]
    # tf.print("Shape after load_image:", tf.shape(image))  # Print shape after loading
    return image, label

def load_image_test(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    image = image / 255.0  # Normalize to [0, 1]
    image = tf.reshape(image, [-1])  # Flatten the test image into a 1D vector
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
print(f"Size of training dataset: {len(train_ds)}")

# Apply transformations
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
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


model = models.Sequential([
    layers.Input(shape=(INPUT_SHAPE,)),
    layers.Dense(8000),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.1),

    # layers.Dense(7000),
    # layers.BatchNormalization(),
    # layers.ReLU(),
    # layers.Dropout(0.3),
    #
    layers.Dense(3000),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.1),
    # #
    # layers.Dense(250),
    # layers.BatchNormalization(),
    # layers.ReLU(),
    # layers.Dropout(0.3),
    # #
    layers.Dense(500),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.1),

    layers.Dense(125),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Dropout(0.1),

    layers.Dense(N_CLASSES, activation='softmax'),
])



try:
    best_model_path = OUTPUT_PATH + "model_epoch_.keras"
    model = load_model(best_model_path)
    print("-----Model Loaded-----")
    start_epoch = 51
except:
    print("-----Model Not Loaded-----")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=939,
        decay_rate=0.95,
        staircase=False
    )
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    start_epoch = 0

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("USING GPU")
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True
)


checkpoint_callback = ModelCheckpoint(
    OUTPUT_PATH+'model_epoch_{epoch:02d}.keras',
    monitor='val_accuracy',
    save_weights_only=False,
    save_best_only=True,
    verbose=1
)

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(labels.numpy())

all_labels = np.vstack(all_labels)
all_labels_indices = np.argmax(all_labels, axis=1)
num_classes = len(np.unique(all_labels_indices))
class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=all_labels_indices)
class_weight_dict = {int(k): float(v) for k, v in enumerate(class_weights)}
print(class_weight_dict)

history = model.fit(
    train_ds,
    initial_epoch = start_epoch,
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







