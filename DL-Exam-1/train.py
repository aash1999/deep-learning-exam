import pandas as pd
import numpy as np
import os
import ast
import warnings
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Reshape, Dropout
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

EXCEL_PATH = "./Exam1-v7/excel/train_test.xlsx"
IMAGES_PATH = "./Exam1-v7/Data/"
OUTPUT_PATH = '/home/ubuntu/DL-Exam-1/DAY4_4_promising_1/'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

BATCH_SIZE = 125
IMG_H = 75
IMG_W = 75
N_CLASSES = 10
EPOCHS = 200

INPUT_SHAPE = IMG_H*IMG_W*3

data_df = pd.read_excel(EXCEL_PATH)
print(data_df.head())
print("Number of classes : ", data_df["target"].nunique())

train_df = data_df[data_df['split'] == 'train']
test_df = data_df[data_df['split'] == 'test']

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = train_df.sample(frac=1, random_state=123).reset_index(drop=True)

test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)


train_paths = [IMAGES_PATH+img_id for img_id in train_df['id'].values]
test_paths = [IMAGES_PATH+img_id for img_id in test_df['id'].values]
train_labels = train_df['target_class'].apply(ast.literal_eval).tolist()
test_labels = test_df['target_class'].apply(ast.literal_eval).tolist()

def load_image(image_path, label):
    # Read and decode the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    # Expr  and not need in test
    image = tf.image.random_flip_left_right(image)  # Random horizontal flip
    image = tf.image.random_flip_up_down(image)  # Random vertical flip
    image = tf.image.random_brightness(image, max_delta=0.1)  # Random brightness change
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)  # Random contrast change
    image = tf.image.random_hue(image, max_delta=0.1)  # Random hue adjustment
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)  # Random saturation change

    # <- End
    # image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
    image = image / 255.0  # Normalize to [0, 1]



    rotated_image_90 = tf.image.rot90(image)

    # Flatten and return the slices for both original and rotated images
    original_slices = tf.split(image, IMG_H, axis=0)  # Split along height
    rotated_slices = tf.split(rotated_image_90, IMG_H, axis=0)  # Split along height

    # Flatten each slice to ensure it has the shape (IMG_W,)
    # original_slices = [tf.squeeze(slice_, axis=0) for slice_ in original_slices]
    # rotated_slices = [tf.squeeze(slice_, axis=0) for slice_ in rotated_slices]

    original_slices = [tf.reshape(tf.squeeze(slice_, axis=0), (-1,)) for slice_ in original_slices]
    rotated_slices = [tf.reshape(tf.squeeze(slice_, axis=0), (-1,)) for slice_ in rotated_slices]

    return tuple(original_slices + rotated_slices), label


def load_image_test(image_path, label):
    # Read and decode the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])

    # image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
    image = image / 255.0  # Normalize to [0, 1]

    rotated_image_90 = tf.image.rot90(image)

    # Flatten and return the slices for both original and rotated images
    original_slices = tf.split(image, IMG_H, axis=0)  # Split along height
    rotated_slices = tf.split(rotated_image_90, IMG_H, axis=0)  # Split along height

    # Flatten each slice to ensure it has the shape (IMG_W,)
    # original_slices = [tf.squeeze(slice_, axis=0) for slice_ in original_slices]
    # rotated_slices = [tf.squeeze(slice_, axis=0) for slice_ in rotated_slices]

    original_slices = [tf.reshape(tf.squeeze(slice_, axis=0), (-1,)) for slice_ in original_slices]
    rotated_slices = [tf.reshape(tf.squeeze(slice_, axis=0), (-1,)) for slice_ in rotated_slices]

    return tuple(original_slices + rotated_slices), label


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

for image, label in train_ds.take(1):
    print(image.shape, label.shape)

# Apply transformations
train_ds = train_ds.map(lambda x, y: load_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: load_image_test(x, y), num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle, batch, and prefetch
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_parallel_mlp(output_dimension, IMG_H, IMG_W):
    num_vectors = IMG_H  # Number of parallel input vectors (72)
    vector_size = IMG_W*3  # Size of each input vector (IMG_W)

    # Define the inputs for both original and rotated images
    original_inputs = [Input(shape=(vector_size,)) for _ in range(num_vectors)]
    rotated_inputs = [Input(shape=(vector_size,)) for _ in range(num_vectors)]  # Rotated images

    # Process the original images through MLP
    processed_vectors_1 = []
    for inp in original_inputs:
        x = Dense(125, activation='relu')(inp)
        x = Dropout(0.3)(x)  # <- Expremintal  (should reach val_acc > 22.5%
        x = Dense(54, activation='relu')(x)
        processed_vectors_1.append(x)

    # Process the rotated images through MLP
    processed_vectors_2 = []
    for inp in rotated_inputs:
        x = Dense(125, activation='relu')(inp)
        x = Dropout(0.3)(x) #<- Expremintal  (should reach val_acc > 22.5%
        x = Dense(54, activation='relu')(x)
        processed_vectors_2.append(x)

    # Concatenate the outputs from both sets of MLPs (original and rotated)
    merged = Concatenate()(processed_vectors_1 + processed_vectors_2)

    # Feed into a dense layer of size 1024
    x = Dense(524, activation='relu')(merged)
    x = Dropout(0.3)(x)
    # Additional dense layer of size 524
    x = Dense(125, activation='relu')(x)

    # Final output layer with softmax activation
    outputs = Dense(output_dimension, activation='softmax')(x)

    model = Model(inputs=original_inputs + rotated_inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Example usage
output_dimension = 10  # Example: 10 output classes
model = build_parallel_mlp(output_dimension,IMG_H, IMG_W)
print(model.summary())

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







