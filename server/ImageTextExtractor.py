import cv2
import numpy as np
import os
from typing import List, Tuple
from CharImageProcessor import CharImageProcessor
from CharacterDetector import CharacterDetector
from ImageRegionProcessor import ImageRegionProcessor
from WordRegionDetector import WordRegionDetector
from structs import CharInfo, WordInfo


# Main class for extracting text from images
class ImageTextExtractor:
    # Initialize the ImageTextExtractor with necessary components and settings
    def __init__(self, image_bytes, char_recognizer, output_dir="extracted_regions"):
        # Create an ImageProcessor object with the given image bytes
        self.binary_region_bboxes = None
        self.binary_regions = None
        self.image_processor = ImageRegionProcessor(image_bytes)
        # Identifies char region
        self.char_detector = CharacterDetector()
        # Create a RegionExtractor object with the ImageProcessor and no initial binary mask
        self.region_extractor = WordRegionDetector(self.image_processor, None)
        # Recognizer the chars from the char regions
        self.char_recognizer = char_recognizer
        # Set the output directory for extracted regions

        self.output_dir = output_dir
        # Initialize visualization flag
        self.show_visualizations = False
        # Initialize flag for saving region images
        self.save_region_images = False

    # Main processing method
    def process(self):
        # Detect char regions
        self.get_binary_char_regions()
        # Combine detected regions
        self.combine_char_to_word_regions()
        # Extract contained regions
        self.match_char_regions_to_word_region()
        # Predict characters in the extracted regions
        self.predict_chars()
        # Save and visualize the results
        self._save_and_visualize_results()
        # Return the list of contour information
        return self.word_info_list

    # detect initial regions in the image
    def get_char_regions(self):
        # Detect regions using the CharacterDetector
        regions = self.char_detector.detect_regions(self.image_processor.gray)
        # Detect valid characters from the regions
        return self.char_detector.detect_valid_chars(regions)

    # detect binary regions
    def get_binary_char_regions(self):
        char_regions = self.get_char_regions()
        # Create a binary mask from the detected regions
        binary_mask = self.image_processor.create_binary_mask(char_regions)
        # Set the binary mask for the region extractor
        self.region_extractor.binary_mask = binary_mask
        # Detect regions in the binary mask
        self.binary_regions = self.char_detector.detect_regions(binary_mask)
        # Compute bounding boxes for the binary regions
        self.binary_region_bboxes = [cv2.boundingRect(region) for region in self.binary_regions]

        # Visualize the results if show_visualizations is True
        if self.show_visualizations:
            # Show the binary mask
            self.image_processor.show_image(binary_mask, "Binary Mask")
            # Create a color mask from the binary mask
            color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            # Draw rectangles around the detected regions
            for bbox in self.binary_region_bboxes:
                x, y, w, h = bbox
                cv2.rectangle(color_mask, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Show the color mask with highlighted regions
            self.image_processor.show_image(color_mask, "New Regions Highlighted")
        return

    # combine detected regions
    def combine_char_to_word_regions(self):
        # Compute convex hulls(contour) for the binary regions
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in self.binary_regions]
        # Create empty image for  combined mask
        combined_mask = np.zeros(self.image_processor.gray.shape, dtype=np.uint8)
        # Draw the hulls on the combined mask
        cv2.drawContours(combined_mask, hulls, -1, (255), -1)
        # Process the combined mask
        word_regions_image = self.image_processor.create_word_regions(combined_mask)
        # Find contours in the closed mask
        self.word_regions = self.image_processor.find_word_contours(word_regions_image)
        # Compute bounding boxes for the combined contours
        self.word_region_bboxes = [cv2.boundingRect(contour) for contour in self.word_regions]
        # Visualize the combined regions
        self._visualize_combined_regions()

    # predict characters in the extracted regions
    def predict_chars(self):
        # Initialize lists to store all images and their corresponding region information
        all_images = []
        region_map = []
        # get all char images to send as a batch to predict
        for contour_idx, contour_info in enumerate(self.word_info_list):
            for region_idx, region_info in enumerate(contour_info.regions):
                # Add the region image to the list of all images
                all_images.append(region_info.image)
                # Store the contour and region indices
                region_map.append((contour_idx, region_idx))

        # Perform batch prediction on all images
        predicted_chars = self.char_recognizer.predict_batch(all_images)

        # Assign predicted characters back to RegionInfo objects
        for (contour_idx, region_idx), predicted_char in zip(region_map, predicted_chars):
            self.word_info_list[contour_idx].regions[region_idx].predicted_char = predicted_char[0]

    # extract contained regions
    def match_char_regions_to_word_region(self):
        # Create an R-tree index for the combined bounding boxes
        self.region_extractor.create_rtree(self.word_region_bboxes)
        # Find regions contained within the combined bounding boxes
        bbox_to_regions = self.region_extractor.find_contained_regions(self.binary_region_bboxes)

        # Initialize the list to store extracted word info
        self.word_info_list = []
        # Iterate through the bbox_to_regions dictionary
        for bbox_id, region_bboxes in bbox_to_regions.items():
            # Get the container bounding box
            container_bbox = self.region_extractor.word_bboxes[bbox_id]
            regions_info = []
            direction = 0
            if len(region_bboxes)<2:
                direction =0
            else:
                if abs(region_bboxes[0][1] - region_bboxes[1][1]) > abs(region_bboxes[0][0] - region_bboxes[1][0]):
                    direction = 1
                else:
                    direction = 2
            # Determine if regions are side by side or stacked
            #is_side_by_side = all(abs(region_bboxes[0][1] - bbox[1]) < 5 for bbox in region_bboxes[1:])

            # Extract each region from the binary mask
            for region_bbox in region_bboxes:
                # Expand the region_bbox based on orientation
                if direction == 2:
                    # Regions are side by side, expand vertically
                    expanded_bbox = (
                        region_bbox[0],  # x
                        container_bbox[1],  # y (use container's y)
                        region_bbox[2],  # w
                        container_bbox[3]  # h (use container's height)
                    )
                if direction == 1:
                    # Regions are stacked, expand horizontally
                    expanded_bbox = (
                        container_bbox[0],  # x (use container's x)
                        region_bbox[1],  # y
                        container_bbox[2],  # w (use container's width)
                        region_bbox[3]  # h
                    )
                if direction == 0:
                    expanded_bbox = region_bbox
                # Extract the region using the expanded bbox
                region_image = self.region_extractor.extract_region_from_binary_mask(expanded_bbox)
                regions_info.append(CharInfo(expanded_bbox, region_image))
            # Add the contour information to the list
            if len(regions_info) > 1:
                self.word_info_list.append(WordInfo(bbox_id, container_bbox, regions_info))




    # visualize combined regions
    def _visualize_combined_regions(self):
        # Skip visualization if show_visualizations is False
        if not self.show_visualizations:
            return
        # Create a copy of the original image
        result = self.image_processor.img.copy()
        # Draw the combined contours on the image
        cv2.drawContours(result, self.word_regions, -1, (0, 255, 0), 2)
        # Draw rectangles for the combined bounding boxes
        for bbox in self.word_region_bboxes:
            x, y, w, h = bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Show the image with combined regions
        if self.show_visualizations:
            self.image_processor.show_image(result, "Combined Regions")

        # Create a debug image
        debug_image = self.image_processor.img.copy()
        # Draw rectangles for the binary region bounding boxes
        for bbox in self.binary_region_bboxes:
            x, y, w, h = bbox
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Draw contours and bounding boxes for the combined regions
        for contour, bbox in zip(self.word_regions, self.word_region_bboxes):
            cv2.drawContours(debug_image, [contour], 0, (0, 0, 255), 2)
            x, y, w, h = bbox
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Show the debug image
        self.image_processor.show_image(debug_image, "Regions and Contours")

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


    # save and visualize results
    def _save_and_visualize_results(self):
        image_processor = CharImageProcessor()
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        # Iterate through all contour information
        for contour_info in self.word_info_list:
            x, y, w, h = contour_info.bounding_box
            print(
                f"Contour {contour_info.word_id} contains {len(contour_info.regions)} regions bounding box: x={x}, y={y}, w={w}, h={h}")
            # Iterate through all regions in the contour
            for i, region_info in enumerate(contour_info.regions):
                print(f"  Region {i}: Predicted character: {region_info.predicted_char}")
                # Save the region image if save_region_images is True
                if self.save_region_images:
                    cv2.imwrite(
                        f"{self.output_dir}/contour_{contour_info.word_id}_region_{i}_{region_info.predicted_char}.png",
                        region_info.image)
                           # Convert to grayscale if it's not already
                    p_image = region_info.image.copy()
                    if len(p_image.shape) == 3 and p_image.shape[2] == 3:
                        p_image = cv2.cvtColor(p_image, cv2.COLOR_BGR2GRAY)

                    # Resize to target size and keep image ratio
                    p_image = self.resize_and_pad(p_image)

                    cv2.imwrite(
                        f"{self.output_dir}_pre/contour_{contour_info.word_id}_region_{i}_{region_info.predicted_char}.png",
                        p_image)
        # Print a message if region images were saved
        if self.save_region_images:
            print(f"Extracted regions have been saved in the '{self.output_dir}' directory.")

        # Visualize individual regions if show_visualizations is True
        if self.show_visualizations:
            self._visualize_individual_regions()

    # visualize individual regions
    def _visualize_individual_regions(self):
        # Iterate through all contour information
        for contour_info in self.word_info_list:
            # Create a copy of the original image
            contour_image = self.image_processor.img.copy()
            x, y, w, h = contour_info.bounding_box
            # Draw a rectangle for the contour bounding box
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Iterate through all regions in the contour
            for region_info in contour_info.regions:
                x, y, w, h = region_info.bounding_box
                # Draw a rectangle for the region bounding box
                cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Add the predicted character as text
                #cv2.putText(contour_image, region_info.predicted_char, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 0, 0), 1)
            # Show the image with contour and region information
            self.image_processor.show_image(contour_image,
                                            f"Contour {contour_info.word_id} with its regions and predictions")

    # set the visualization flag
    def set_visualization(self, show_visualizations: bool):
        self.show_visualizations = show_visualizations

    # set the flag for saving region images
    def set_save_region_images(self, save_region_images: bool):
        self.save_region_images = save_region_images