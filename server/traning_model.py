import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, namedtuple, Counter
import pickle


class ModelTrainer:
    def __init__(self, main_folders, main_char_sequence, img_height=64, img_width=64, batch_size=32):
        self.main_folders = defaultdict(str, main_folders)
        self.main_char_sequence = defaultdict(str, main_char_sequence)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_classes = sum(len(seq) for seq in self.main_char_sequence.values())
        self.label_encoder = LabelEncoder()
        self.ImageData = namedtuple('ImageData', ['path', 'label'])
        self.valid_extensions = {'.png', '.jpg', '.jpeg'}

    def load_and_preprocess_data(self):
        file_paths, labels = self._load_image_data()
        integer_labels = self.label_encoder.fit_transform(labels)

        data = [self.ImageData(path, label) for path, label in zip(file_paths, integer_labels)]
        np.random.shuffle(data)

        split = int(len(data) * 0.8)
        train_data = data[:split]
        test_data = data[split:]

        self.label_distribution = Counter(labels)

        self.data_df = pd.DataFrame(data, columns=['path', 'label'])

        self.train_dataset = self._create_dataset(train_data)
        self.test_dataset = self._create_dataset(test_data, repeat=False)
        self.steps_per_epoch = len(train_data) // self.batch_size

    def _load_image_data(self):
        file_paths, labels = [], []
        for folder, label_sequence in zip(self.main_folders.values(), self.main_char_sequence.values()):
            for subfolder, sublabel in zip(next(os.walk(folder))[1], label_sequence):
                subfolder_path = os.path.join(folder, subfolder)
                image_files = [f for f in os.listdir(subfolder_path) if
                               os.path.splitext(f)[1].lower() in self.valid_extensions]
                file_paths.extend([os.path.join(subfolder_path, f) for f in image_files])
                labels.extend([sublabel] * len(image_files))
        return file_paths, labels

    def _preprocess_image(self, file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        return img, label

    def _create_dataset(self, data, repeat=True):
        file_paths, labels = zip(*[(d.path, d.label) for d in data])
        dataset = tf.data.Dataset.from_tensor_slices((list(file_paths), list(labels)))
        dataset = dataset.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        if repeat:
            dataset = dataset.repeat()
        return dataset

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.img_height, self.img_width, 1))
        x = inputs
        for filters in [32, 64, 128, 256]:
            x = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',#better for intger numbers and many categories
                           metrics=['accuracy'])

    def train(self, epochs=50):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.test_dataset,
            callbacks=[early_stopping]
        )
        return history

    def save_model(self, filename):
        self.model.save(f"{filename}.keras")
        self.model.save(f"{filename}.h5")

    def save_label_encoder(self, filename):
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)


# Usage example
if __name__ == "__main__":
    main_folders = {
        "uppercase_english": r"C:\see_and_seek\create_data\data\capital_english_letters",
        "lowercase_english": r"C:\see_and_seek\create_data\data\english_lowercase_letters",
    }

    # Define the labels corresponding to each main folder
    main_char_sequence = {
        "uppercase_english": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "lowercase_english": "abcdefghijklmnopqrstuvwxyz",
    }

    trainer = ModelTrainer(main_folders, main_char_sequence)
    trainer.load_and_preprocess_data()
    trainer.build_model()
    history = trainer.train()
    trainer.save_model("character_model_english_2")
    trainer.save_label_encoder("label_encoder_english_2")

    # Optional: Print some statistics
    print("Label distribution:", trainer.label_distribution)
    print("Data summary:\n", trainer.data_df.describe())