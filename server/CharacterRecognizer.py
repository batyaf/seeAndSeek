import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from CharImageProcessor import CharImageProcessor
import time

# Class for character recognition
class CharacterRecognizer:
    def __init__(self, model_path, characters):
        # Load the model from the specified path safely
        self.model = self.load_model_safe(model_path)
        # Initialize the label encoder and fit it with the provided characters
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(characters)
        self.target_size = (64, 64)
        # Initialize the image processor
        self.image_processor = CharImageProcessor()

    # Load the model safely, handling any exceptions
    def load_model_safe(self, model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # Preprocess the character image for model prediction
    def pre_process_for_model(self, char_image):
        return self.image_processor.preprocess_image(char_image)


    def resize_and_pad(self,img,size =(64,64)):
        h, w = img.shape[:2]
        aspect_ratio = w / h

        # Calculate new dimensions while maintaining aspect ratio
        if aspect_ratio > 1:
            new_w = size[0]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = size[1]
            new_w = int(new_h * aspect_ratio)

        # Resize the image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a new image with the desired size and background color
        new_img = np.full((size[1], size[0]), 255, dtype=np.uint8)

        # Calculate position to paste the resized image
        x_offset = (size[0] - new_w) // 2
        y_offset = (size[1] - new_h) // 2

        # Paste the resized image onto the new image
        new_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return new_img

    def preprocess_image(self, image):
        # Convert to grayscale if it's not already
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to target size and keep image ratio
        #image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        image = self.resize_and_pad(image)
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        # Add channel dimension
        image = np.expand_dims(image, axis=-1)

        return image


    # # Predict the character from the image
    # def predict_char(self, images):
    #     # Preprocess all images in the batch
    #     processed_images = [self.preprocess_image(img) for img in images]

    #     # Stack preprocessed images into a single numpy array
    #     batch = np.stack(processed_images, axis=0)

    #     # Perform batch prediction
    #     predictions = self.model.predict(batch)
    #     # Find the index of the highest probability prediction
    #     pred_char_index = np.argmax(predictions, axis=-1)[0]
    #     try:
    #         # Convert the predicted index back to a character
    #         pred_char = self.label_encoder.inverse_transform([pred_char_index])[0]
    #     except ValueError:
    #         # If conversion fails, return a question mark
    #         pred_char = '?'
    #     # Return the predicted character
    #     return pred_char

    def predict_batch(self, images):
        #input_shape = self.model.input_shape
        #print(f"Expected input shape: {input_shape}")
        # Preprocess all images in the batch
        processed_images = [self.preprocess_image(img) for img in images]
        # Stack preprocessed images into a single numpy array
        batch = np.stack(processed_images, axis=0)
        predictions = self.model.predict(batch)

        # Convert predictions to character labels
        predicted_chars = [self.label_encoder.inverse_transform([np.argmax(pred)]) for pred in predictions]

        return predicted_chars

    def visual_predicted_char(self, char_image, pred_char):
        # Visualize the predicted character
        plt.figure(figsize=(4, 4))
        plt.imshow(char_image, cmap='gray')
        plt.title(f"Predicted: {pred_char}")
        plt.axis('off')
        plt.show()
