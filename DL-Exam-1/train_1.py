import pandas as pd
import numpy as np
import os
import ast
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print("USING GPU")
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Suppress TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Paths and Constants
EXCEL_PATH = "./Exam1-v7/excel/train_test.xlsx"
IMAGES_PATH = "./Exam1-v7/Data/"
OUTPUT_PATH = '/home/ubuntu/DL-Exam-1/DAY7_1/'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

BATCH_SIZE = 125
IMG_H, IMG_W = 125, 125  # Image Dimensions
N_CLASSES = 10
EPOCHS =100
INPUT_SHAPE = (IMG_H, IMG_W, 3)

# Load Data
data_df = pd.read_excel(EXCEL_PATH)
print(data_df.head())
print("Number of classes : ", data_df["target"].nunique())
from sklearn.model_selection import train_test_split
# Split Data
train_df = data_df.sample(frac=1, random_state=123).reset_index(drop=True)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,   # 20% for validation
    stratify=train_df['target'],  # Maintain class distribution
    random_state=42
)

test_df = data_df[data_df['split'] == 'test'].sample(frac=1, random_state=42).reset_index(drop=True)


# File Paths and Labels
def process_labels(labels):
    return np.array(ast.literal_eval(labels))


train_df['target_class'] = train_df['target_class'].apply(process_labels)
val_df['target_class'] = val_df['target_class'].apply(process_labels)
test_df['target_class'] = test_df['target_class'].apply(process_labels)

# Image Data Generator
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
import cv2
def preprocess_noisy_image(img):
    # Add noise reduction or other image preprocessing steps here
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Example: Gaussian Blur for noise reduction
    return img

train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,  # Random rotations between 0 and 40 degrees
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Random shearing transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest',  # Strategy for filling missing pixels after transformation
    # preprocessing_function=preprocess_noisy_image,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_dataframe(

    train_df,
    workers=20,
    use_multiprocessing=True,
    directory=IMAGES_PATH,
    x_col='id',
    y_col='target',
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',

)

val_generator = train_datagen.flow_from_dataframe(
    val_df,
    directory=IMAGES_PATH,
    x_col='id',
    y_col='target',
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    workers=20,  # Move this here
    use_multiprocessing=True,  # Move this here
    max_queue_size=200  # Move this here
)

test_generator = datagen.flow_from_dataframe(
    test_df,
    directory=IMAGES_PATH,
    x_col='id',
    y_col='target',
    target_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    workers=20,  # Move this here
    use_multiprocessing=True,  # Move this here
    max_queue_size=20  # Move this here
)

from tensorflow.keras.optimizers import Adam
# Define MLP Model
def build_mlp_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(524)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    initial_learning_rate = 0.0001
    # optimizer = Adam(learning_rate=initial_learning_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Build Model
model = build_mlp_model(INPUT_SHAPE, N_CLASSES)
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_PATH, 'best_model.keras'), monitor='train_loss', save_best_only=True,
                             mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
class_weight_dict = {0: 0.44161370804500305, 6: 0.7705930671923772, 3: 0.9078873512048776,
                     1: 0.9155963302752294, 7: 1.0121048656813032, 5: 1.0321487512377598, 9: 1.3701182999853951,
                     8: 1.574290988420876, 2: 2.0500874125874127, 4: 2.3015701668302255}


def generator_to_dataset(generator, batch_size):
    # Fetch the data from the generator
    def gen():
        for x, y in generator:
            yield x, y

    # Create a tf.data.Dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, IMG_H, IMG_W, 3), dtype=tf.float32),  # Shape of images
            tf.TensorSpec(shape=(batch_size, N_CLASSES), dtype=tf.float32)  # Shape of labels
        )
    )
    return dataset
train_dataset = generator_to_dataset(train_generator, BATCH_SIZE)
val_dataset = generator_to_dataset(val_generator, BATCH_SIZE)

# Apply prefetch for better performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE).cache()
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE).cache()

PLOTS_DIR = os.path.join(OUTPUT_PATH, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Train Model
with tf.device('/GPU:0'):
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,

        callbacks=[checkpoint, early_stop]
    )

# Evaluate Model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Save Model
model.save(os.path.join(OUTPUT_PATH, 'final_mlp_model.keras'))


import matplotlib.pyplot as plt
import os

def save_training_plots(history, out_path):
    plots_dir = os.path.join(out_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.savefig(os.path.join(plots_dir, "accuracy_plot.png"))
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.savefig(os.path.join(plots_dir, "loss_plot.png"))
    plt.close()


import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(model, test_generator, out_path):
    plots_dir = os.path.join(out_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get predictions
    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices, yticklabels=test_generator.class_indices)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()

save_training_plots(history, PLOTS_DIR)

# Save confusion matrix using the test dataset
save_confusion_matrix(model, test_generator, PLOTS_DIR)