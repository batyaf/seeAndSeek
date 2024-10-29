import cv2
import numpy as np

class ImageRegionProcessor:
    def __init__(self, image_bytes):
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the numpy array to OpenCV image format
        self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Rotate the image 90 degrees clockwise
        self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        # Convert the image to grayscale
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Get the height and width of the image
        self.height, self.width = self.img.shape[:2]

    def show_image(self, image, title):
        # Display the image using matplotlib
        plt.figure(figsize=(20, 12))
        if len(image.shape) == 2:
            # If the image is grayscale
            plt.imshow(image, cmap='gray')
        else:
            # If the image is color, convert from BGR to RGB for display
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()

    # Create a binary mask based on the provided regions
    def create_binary_mask(self, regions):
        binary_mask = np.ones(self.gray.shape, dtype=np.uint8) * 255
        for region in regions:
            for point in region:
                binary_mask[point[1], point[0]] = 0
        return binary_mask

    # Process image to blue char edges this will cause close char region to connect to one region
    def create_word_regions(self, mask):
        # Process the mask using morphological operations
        kernel = np.ones((5,5), np.uint8)
        # Function that expands the char regions
        dilated_mask = cv2.dilate(mask, kernel, iterations=2)
        # Connects close regions
        closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)
        return closed_mask

    def find_word_contours(self, mask):
        # Find contours in the mask
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
