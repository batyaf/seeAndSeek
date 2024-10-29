import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define the main folders containing subfolders with images
main_folders = {
    "printed_hebrew": r"C:\see_try\data\hebrew_typefaces"
}

# Define the labels corresponding to each main folder
main_char_sequence = {
    "printed_hebrew": "אבחדעגהךכקלםמןנףפרסשתטויצץז"
}

def load_and_preprocess_images(main_folders, main_char_sequence):
    file_paths = []
    labels = []

    for folder, label in zip(main_folders.values(), main_char_sequence.values()):
        for subfolder, sublabel in zip(next(os.walk(folder))[1], label):
            subfolder_path = os.path.join(folder, subfolder)
            # Check if there are image files in the current subfolder
            image_files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            if image_files:
                # Load and preprocess images
                for file in image_files:
                    image_path = os.path.join(subfolder_path, file)
                    file_paths.append(image_path)
                    labels.append(sublabel)

    return file_paths, labels

# Define a function to read and preprocess the images
def preprocess_image(file_path, file_label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)  # Read grayscale images with 1 channel
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img, file_label

num_classes = 27
img_height = 64
img_width = 64

file_paths, labels = load_and_preprocess_images(main_folders, main_char_sequence)
# Convert string labels to integer labels
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)

# Combine file paths and labels into a single list
data = list(zip(file_paths, integer_labels))

# Shuffle the data
np.random.shuffle(data)

# Split the data into training and testing sets
split = int(len(data) * 0.8)  # 80% training, 20% testing
train_data = data[:split]
test_data = data[split:]

# Unzip the training and testing data
train_file_paths, train_labels = zip(*train_data)
test_file_paths, test_labels = zip(*test_data)

print("Shape of train_file_paths:", np.array(train_file_paths).shape)
print("Shape of train_labels:", np.array(train_labels).shape)

# Define the generator function
def generator(file_paths, labels):
    for path, label in zip(file_paths, labels):
        yield path, label

# Create datasets for training and testing
# Assuming you have a variable `epochs` with the desired number of epochs
num_epochs = epochs

# Create datasets for training and testing
train_dataset = tf.data.Dataset.from_generator(
    generator,
    args=[train_file_paths, train_labels],
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
).repeat().take(len(train_data) * num_epochs)  # Limit the number of repetitions

test_dataset = tf.data.Dataset.from_generator(
    generator,
    args=[test_file_paths, test_labels],
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
)

# Map the preprocess function to the datasets and batch them
batch_size = 32
train_dataset = train_dataset.map(preprocess_image).batch(batch_size)
test_dataset = test_dataset.map(preprocess_image).batch(batch_size)

# Define your model with batch normalization using the functional API
inputs = tf.keras.layers.Input(shape=(img_height, img_width, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model1 = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model with a larger learning rate
model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Train the model using model.fit and the early stopping callback
steps_per_epoch = len(train_data) // batch_size
epochs = 50
history1 = model1.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_dataset,
    callbacks=[early_stopping]
)

model1.save("hebrew_model.keras")
model1.save("hebrew_model.h5")