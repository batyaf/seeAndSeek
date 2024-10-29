import cv2


# Class for detecting and validating character regions in an image
class CharacterDetector:
    # Initialize the CharacterDetector with MSER parameters
    def __init__(self):
        # Create an MSER (Maximally Stable Extremal Regions) detector
        self.mser = cv2.MSER_create()
        # Set the delta parameter for the MSER detector
        self.mser.setDelta(5)
        # Set the minimum area for detected regions
        self.mser.setMinArea(50)
        # Set the maximum area for detected regions
        self.mser.setMaxArea(10000)

    # Detect regions in a grayscale image using MSER
    def detect_regions(self, gray_image):
        # Detect regions in the grayscale image using MSER
        regions, _ = self.mser.detectRegions(gray_image)
        # Return the detected regions
        return regions

    # Check if a given region meets the criteria for a valid character
    def is_valid_char(self, region):
        # Get the bounding rectangle of the region
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

        # Use the smaller dimension for aspect ratio calculation
        aspect_ratio = float(min(w, h)) / max(w, h)

        # Calculate the area of the bounding rectangle
        char_area = w * h

        # Calculate the extent (ratio of contour area to bounding rectangle area)
        extent = float(cv2.contourArea(region)) / char_area

        # Define minimum and maximum aspect ratios
        min_aspect_ratio, max_aspect_ratio = 0.2, 1.0

        # Define minimum and maximum character areas
        min_char_area, max_char_area = 50, 10000

        # Define minimum extent
        min_extent = 0.3

        # Check if the region meets all criteria for a valid character
        return (min_aspect_ratio < aspect_ratio < max_aspect_ratio and
                min_char_area < char_area < max_char_area and
                extent > min_extent)


    # Filter and return only the valid character regions
    def detect_valid_chars(self, regions):
        # Filter regions to only include those that are valid characters
        return [region for region in regions if self.is_valid_char(region)]