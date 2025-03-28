import pandas as pd
import numpy as np
import os
import ast
import warnings
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Reshape, Dropout, LeakyReLU, Add, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

EXCEL_PATH = "./Exam1-v7/excel/train_test.xlsx"
IMAGES_PATH = "./Exam1-v7/Data/"
OUTPUT_PATH = '/home/ubuntu/DL-Exam-1/DAY5/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

BATCH_SIZE = 300
IMG_H = 100
IMG_W = 100
N_CLASSES = 10
EPOCHS = 400

INPUT_SHAPE = IMG_H*IMG_W*3


def random_rotate(image):
    if len(image.shape) != 3:
        raise ValueError("Image must be a 3D tensor with shape (height, width, channels)")
    angle = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=angle)
    return image


def random_zoom(image, zoom_range=(0.8, 1.2)):
    """Randomly zoom in or out on an image."""
    zoom_factor = tf.random.uniform([], minval=zoom_range[0], maxval=zoom_range[1], dtype=tf.float32)
    original_shape = tf.shape(image)
    new_height = tf.cast(original_shape[0], tf.float32) * zoom_factor
    new_width = tf.cast(original_shape[1], tf.float32) * zoom_factor
    image = tf.image.resize(image, [new_height, new_width])
    image = tf.image.resize_with_crop_or_pad(image, target_height=original_shape[0], target_width=original_shape[1])

    return image

def random_blackout(image):
    shape = tf.shape(image)
    h, w = shape[0], shape[1]  # Extract height & width
    blackout_h, blackout_w = 10, 10  # Blackout patch size

    # Ensure blackout does not go out of bounds
    top = tf.random.uniform([], 0, h - blackout_h, dtype=tf.int32)
    left = tf.random.uniform([], 0, w - blackout_w, dtype=tf.int32)

    # Create a mask filled with ones
    mask = tf.ones_like(image)

    # Zero-out the selected blackout region
    blackout_region = tf.zeros((blackout_h, blackout_w, 3), dtype=image.dtype)
    mask = tf.tensor_scatter_nd_update(
        mask,
        indices=tf.stack(tf.meshgrid(tf.range(top, top + blackout_h), tf.range(left, left + blackout_w), indexing='ij'), axis=-1),
        updates=blackout_region
    )

    return image * mask  # Apply blackout


def data_aug(image):
    choice = tf.random.uniform([], minval=0, maxval=5, dtype=tf.int32)
    if choice ==1:
        image = tf.image.random_flip_left_right(image)  # Random horizontal flip
        image = tf.image.random_flip_up_down(image)  # Random vertical flip
        image = random_rotate(image)
    elif choice ==2:
        image = tf.image.random_brightness(image, max_delta=0.25)  # Random brightness change
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)  # Random contrast change
    elif choice ==3:
        image = tf.image.random_hue(image, max_delta=0.25)  # Random hue adjustment
        image = tf.image.random_saturation(image, lower=0.2, upper=1.8)  # Random saturation change
    # elif choice == 4:  # Adding white noise
    #     noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.07)  # Adjust stddev as needed
    #     image = tf.clip_by_value(image + noise, 0.0, 1.0)  # Ensure values stay in valid range
    elif choice == 4:
        image = random_zoom(image)
    elif choice == 5:
        image = random_blackout(image)

    return image



import random
# def load_image(image_path, label):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMG_H, IMG_W])
#     image = data_aug(image)
#     image = image / 255.0  # Normalize to [0, 1]
#     # rotated_image_90 = tf.image.rot90(image)
#     original_slices = tf.split(image, IMG_H, axis=0)  # Split along height
#     # rotated_slices = tf.split(rotated_image_90, IMG_H, axis=0)  # Split along heigh
#     original_slices = [tf.reshape(tf.squeeze(slice_, axis=0), (-1,)) for slice_ in original_slices]
#     # rotated_slices = [tf.reshape(tf.squeeze(slice_, axis=0), (-1,)) for slice_ in rotated_slices]
#
#     # return tuple(original_slices + rotated_slices), label
#     return tuple(original_slices), label


def histogram_equalization(image):
    # Convert the image to grayscale
    gray_image = tf.image.rgb_to_grayscale(image)

    # Compute the histogram of the grayscale image
    hist = tf.histogram_fixed_width(tf.cast(gray_image, tf.float32), [0, 255], nbins=256)

    # Compute the cumulative distribution function (CDF)
    cdf = tf.cumsum(hist)
    cdf_min = tf.reduce_min(tf.boolean_mask(cdf, cdf > 0))

    # Compute the equalization lookup table
    cdf_scaled = (cdf - cdf_min) * 255 / (tf.size(cdf) - cdf_min)
    cdf_scaled = tf.clip_by_value(cdf_scaled, 0, 255)
    cdf_scaled = tf.cast(cdf_scaled, tf.uint8)

    # Apply the equalization to the grayscale image
    equalized_image = tf.gather(cdf_scaled, tf.cast(gray_image, tf.int32))

    # Convert back to RGB
    equalized_image = tf.image.grayscale_to_rgb(equalized_image)

    return equalized_image

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    image = data_aug(image)
    # image = histogram_equalization(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0  # Normalize to [0, 1]
    # image = histogram_equalization(image)
    # Flatten the entire image into a 1D vector
    flattened_vector = tf.reshape(image, [-1])  # Shape: (IMG_H * IMG_W * 3,)

    return flattened_vector, label





def load_image_test(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    # image = histogram_equalization(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0  # Normalize to [0, 1]
    # image = histogram_equalization(image)

    # Flatten the entire image into a 1D vector
    flattened_vector = tf.reshape(image, [-1])  # Shape: (IMG_H * IMG_W * 3,)

    return flattened_vector, label

# def load_image_test(image_path, label):
#     # Read and decode the image
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMG_H, IMG_W])
#     image = image / 255.0  # Normalize to [0, 1]
#
#     # rotated_image_90 = tf.image.rot90(image)
#     original_slices = tf.split(image, IMG_H, axis=0)  # Split along height
#     # rotated_slices = tf.split(rotated_image_90, IMG_H, axis=0)  # Split along height
#     original_slices = [tf.reshape(tf.squeeze(slice_, axis=0), (-1,)) for slice_ in original_slices]
#     # rotated_slices = [tf.reshape(tf.squeeze(slice_, axis=0), (-1,)) for slice_ in rotated_slices]
#
#     # return tuple(original_slices + rotated_slices), label
#     return tuple(original_slices), label

data_df = pd.read_excel(EXCEL_PATH)
print(data_df.head())
print("Number of classes : ", data_df["target"].nunique())

from collections import defaultdict


def stratified_batching(image_paths, labels, batch_size):
    """Ensures each batch contains a uniform distribution of target classes."""

    # Group images by class
    class_dict = defaultdict(list)
    for path, label in zip(image_paths, labels):
        class_dict[tuple(label)].append(path)

    # Find number of classes and calculate samples per class per batch
    num_classes = len(class_dict)
    samples_per_class_per_batch = batch_size // num_classes

    # Ensure each batch has an equal number of samples per class
    batches = []
    while all(len(v) >= samples_per_class_per_batch for v in class_dict.values()):
        batch = []
        for label, images in class_dict.items():
            selected_images = images[:samples_per_class_per_batch]
            batch.extend((img, label) for img in selected_images)
            class_dict[label] = images[samples_per_class_per_batch:]  # Remove used samples

        np.random.shuffle(batch)  # Shuffle within the batch
        batches.append(batch)

    # Flatten list and separate paths/labels
    final_paths, final_labels = zip(*[item for batch in batches for item in batch])
    final_labels = [list(lbl) for lbl in final_labels]  # Convert tuple back to list

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((list(final_paths), final_labels))
    dataset = dataset.shuffle(len(final_paths))
    dataset = dataset.map(lambda x, y: load_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_df = data_df[data_df['split'] == 'train']
test_df = data_df[data_df['split'] == 'test']


train_df = train_df.sample(frac=1, random_state=123).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)


train_paths = [IMAGES_PATH+img_id for img_id in train_df['id'].values]
test_paths = [IMAGES_PATH+img_id for img_id in test_df['id'].values]
train_labels = train_df['target_class'].apply(ast.literal_eval).tolist()
test_labels = test_df['target_class'].apply(ast.literal_eval).tolist()

print(train_paths[:10])
print(train_labels[:10])

train_ds = stratified_batching(train_paths, train_labels, BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.shuffle(len(test_paths))
test_ds = test_ds.map(lambda x, y: load_image_test(x, y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
# test_ds = test_ds.map(lambda x, y: load_image_test(x, y), num_parallel_calls=tf.data.AUTOTUNE)
# test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def MLP():
    input_shape = (IMG_H * IMG_W * 3,)  # (12288,) for 64x64x3 images

    # Functional API for MLP Model
    inputs = tf.keras.Input(shape=input_shape)

    # Hidden Layers
    x = Dense(24000, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    x = Dense(6000, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    x = Dense(3000, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    x = Dense(1500, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    x = Dense(750, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    x = Dense(250, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    x = Dense(125, activation='relu')(inputs)

    # Output Layer
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    from tensorflow.keras.optimizers import Adam

    initial_learning_rate = 0.0001  # Set your desired initial learning rate

    # Define the Adam optimizer with the initial learning rate
    optimizer = Adam(learning_rate=initial_learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    return model

from tensorflow.keras.regularizers import l2
def MLP_1():
    input_shape = (IMG_H * IMG_W * 3,)
    inputs = tf.keras.Input(shape=input_shape)

    # U-Shape Increasing Path
    x1 = Dense(124, kernel_regularizer=l2(0.001))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = tf.keras.activations.relu(x1)

    x3 = Dense(10, kernel_regularizer=l2(0.001))(x1)
    x3 = BatchNormalization()(x3)
    x3 = tf.keras.activations.relu(x3)

    x5 = Dense(124, kernel_regularizer=l2(0.001))(x3)
    x5 = BatchNormalization()(x5)
    x5 = Add()([x5, x1])  # Skip connection from 6k layer
    x5 = tf.keras.activations.relu(x5)

    # x6 = Dense(input_shape[0], kernel_regularizer=l2(0.001))(x5)
    # x6 = BatchNormalization()(x6)
    # x6 = Add()([x6, inputs])  # Skip connection from input 20k layer
    # x6 = tf.keras.activations.relu(x6)

    # Additional Layers after U-Shape
    # x7 = Dense(50, kernel_regularizer=l2(0.001))(x5)
    # x7 = BatchNormalization()(x7)
    # x7 = tf.keras.activations.relu(x7)

    x8 = Dense(5, kernel_regularizer=l2(0.001))(x5)
    x8 = tf.keras.activations.relu(x8)

    outputs = Dense(10, activation='softmax')(x8)

    # Define Model
    model = Model(inputs=inputs, outputs=outputs)
    from tensorflow.keras.optimizers import Adam, Nadam

    initial_learning_rate = 0.0001#0.00000003  # Set your desired initial learning rate
    optimizer = Adam(learning_rate=initial_learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=5.0 ) #<- 0.000001
    # Define the Adam optimizer with the initial learning rate
    # optimizer = Nadam(learning_rate=initial_learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    return model


def MLP_2():
    input_shape = (IMG_H * IMG_W * 3,)  # Example flattened image shape (12288,)
    inputs = tf.keras.Input(shape=input_shape)

    # U-Shape Increasing Path
    x1 = Dense(6000, activation='relu')(inputs)
    x1 = Dropout(0.3)(x1)

    x2 = Dense(3000, activation='relu')(x1)
    x2 = Dropout(0.3)(x2)

    x3 = Dense(1024, activation='relu')(x2)
    x3 = Dropout(0.3)(x3)

    # U-Shape Decreasing Path (With Skip Connections)
    x4 = Dense(3000, activation='relu')(x3)
    x4 = Dropout(0.3)(x4)
    x4 = Add()([x4, x2])  # Skip connection from 3k layer

    x5 = Dense(6000, activation='relu')(x4)
    x5 = Dropout(0.3)(x5)
    x5 = Add()([x5, x1])  # Skip connection from 6k layer

    x6 = Dense(input_shape[0], activation='relu')(x5)
    x6 = Dropout(0.3)(x6)
    x6 = Add()([x6, inputs])  # Skip connection from input 20k layer

    # Additional Layers after U-Shape
    x7 = Dense(6000, activation='relu')(x6)
    x7 = Dropout(0.3)(x7)

    x8 = Dense(3200, activation='relu')(x7)
    x8 = Dropout(0.3)(x8)

    x9 = Dense(1024, activation='relu')(x8)
    x9 = Dropout(0.3)(x9)

    x10 = Dense(524, activation='relu')(x9)
    x10 = Dropout(0.3)(x10)

    x11 = Dense(125, activation='relu')(x10)
    x11 = Dropout(0.3)(x11)

    # Output Layer (10 classes)
    outputs = Dense(10, activation='softmax')(x11)

    # Define Model
    model = Model(inputs=inputs, outputs=outputs)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

output_dimension = 10  # Example: 10 output classes
model = MLP_1()
model_size_mb = model.count_params() * 4 / (1024 ** 2)
print(f"Model Size: {model_size_mb:.2f} MB")

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("USING GPU")
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

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


class GradientLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        total_grad = 0.0
        num_vars = 0
        for var in self.model.trainable_variables:
            grad = tf.norm(var)
            total_grad += grad
            num_vars += 1
        avg_grad = total_grad / num_vars
        print(f"Epoch {epoch + 1}: Avg Gradient Norm = {avg_grad.numpy():.6f}")

gradient_logger = GradientLogger()
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, verbose=1)

class SaveEveryNEpochs(tf.keras.callbacks.Callback):
    def __init__(self, save_freq=20, save_path="DAY5/models/model_epoch_{epoch:03d}.keras"):
        super(SaveEveryNEpochs, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:  # Save at every 20th epoch
            save_file = self.save_path.format(epoch=epoch + 1)
            self.model.save(save_file)
            print(f"Model saved at: {save_file}")

# Initialize the callback
save_callback = SaveEveryNEpochs(save_freq=20)

history = model.fit(
    train_ds,
    initial_epoch = 0,
    epochs=EPOCHS,
    validation_data=test_ds,
    class_weight=class_weight_dict,
    verbose=2,
    callbacks=[gradient_logger] #lr_scheduler] #checkpoint_callback
)

model.save(OUTPUT_PATH+'model_Alicia.keras')

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

plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(plots_dir, "accuracy_plot.png"))
plt.close()







