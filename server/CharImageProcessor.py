import cv2
import numpy as np

class CharImageProcessor:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    def add_white_background(self, image):
        height, width = image.shape[:2]
        new_height, new_width = self.target_size

        if height > new_height or width > new_width:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        new_image = np.ones((new_height, new_width), dtype=np.uint8) * 255
        height, width = image.shape[:2]
        offset_x = (new_width - width) // 2
        offset_y = (new_height - height) // 2
        new_image[offset_y:offset_y+height, offset_x:offset_x+width] = image

        return new_image

    def enhance_image(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        enhanced_image = (image - min_val) / (max_val - min_val) * 255
        enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
        return enhanced_image

    def preprocess_image_1(self, image):
         #Check if the image is already grayscale
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        if black_pixels > white_pixels:
            binary = cv2.bitwise_not(binary)
        enhanced_image = self.enhance_image(binary)
        padded_image = cv2.copyMakeBorder(enhanced_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        resized_image = cv2.resize(padded_image, self.target_size)
        preprocessed_image = resized_image.astype(np.float32) / 255.0
        preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)
        return np.expand_dims(preprocessed_region, axis=0)

    def preprocess_image(self, image):
        # Check if the image is already grayscale
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure the image is binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Add padding to make the image square
        h, w = binary.shape
        size = max(h, w)
        padded = np.zeros((size, size), dtype=np.uint8)
        padded[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = binary

        # Resize to target size
        resized = cv2.resize(padded, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0

        # Add channel dimension
        with_channel = np.expand_dims(normalized, axis=-1)
        return with_channel
        # Add batch dimension of 0
        #final = np.expand_dims(with_channel, axis=0)

        # Remove the first dimension to get (0, 64, 64, 1)
        #final = final[0:0]

        return final


    def binarize_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        height, width = binary_image.shape
        frame_pixels = []
        for i in range(width):
            frame_pixels.append(binary_image[0, i])
            frame_pixels.append(binary_image[height-1, i])
        for i in range(height):
            frame_pixels.append(binary_image[i, 0])
            frame_pixels.append(binary_image[i, width-1])
        black_frame_pixels = np.sum(np.array(frame_pixels) == 0)
        white_frame_pixels = np.sum(np.array(frame_pixels) == 255)
        if black_frame_pixels > white_frame_pixels:
            binary_image = cv2.bitwise_not(binary_image)
        return binary_image
