# Import necessary libraries
from rtree import index
import cv2
import numpy as np
from typing import List, Dict, Tuple
from intervaltree import IntervalTree, Interval

# Class for extracting and managing regions in an image
class WordRegionDetector:
    # Initialize the WordRegionDetector with an image processor and binary mask
    def __init__(self, image_processor, binary_mask):
        # Store the image processor
        self.image_processor = image_processor
        # Store the binary mask
        self.binary_mask = binary_mask
        # Initialize R-tree index as None
        self.rtree_idx = None
        # Initialize combined bounding boxes as None
        self.word_bboxes = None

    # Create an R-tree index from combined bounding boxes
    # Helps searching the list of "word" regions when we want to find which combined region contains the char region
    def create_rtree(self, word_bboxes: List[Tuple[int, int, int, int]]) -> None:
        # Store the combined bounding boxes
        self.word_bboxes = word_bboxes
        # Create R-tree property object
        p = index.Property()
        # Set the dimension of the R-tree to 2 (for 2D space)
        p.dimension = 2
        # Create the R-tree index with the specified properties
        self.rtree_idx = index.Index(properties=p)
        # Insert each bounding box into the R-tree
        for i, (x, y, w, h) in enumerate(word_bboxes):
            self.rtree_idx.insert(i, (x, y, x + w, y + h))

    # Find regions contained within the combined bounding boxes
    # Loops all char regions, finds for each char region it's word region (the combined region) and add the region
    # to the list of char regions of the word region
    def find_contained_regions(self, region_bboxes: List[Tuple[int, int, int, int]]) -> Dict[
        int, List[Tuple[int, int, int, int]]]:
        # Initialize a dictionary to store regions for each combined bounding box
        word_char_regions = {i: [] for i in range(len(self.word_bboxes))}


        # Iterate through each region bounding box
        for region_bbox in region_bboxes:
            rx, ry, rw, rh = region_bbox
            # Find potential containing bounding boxes using R-tree
            potential_bboxes = list(self.rtree_idx.intersection((rx, ry, rx + rw, ry + rh)))

            # Check if the region is contained in any of the potential bounding boxes
            for bbox_idx in potential_bboxes:
                container_bbox = self.word_bboxes[bbox_idx]
                if self.is_bbox_contained(region_bbox, container_bbox):
                    word_char_regions[bbox_idx].append(region_bbox)

        # Sort and remove contained regions for each group
        for bbox_idx in word_char_regions:
            word_char_regions[bbox_idx] = self.sort_bboxes(word_char_regions[bbox_idx])
            word_char_regions[bbox_idx] = self.remove_overlaping_regions(word_char_regions[bbox_idx])

        return word_char_regions

    # Static method to check if one bounding box is contained within another bounding box
    @staticmethod
    def is_bbox_contained(inner_bbox: Tuple[int, int, int, int], outer_bbox: Tuple[int, int, int, int]) -> bool:
        ix, iy, iw, ih = inner_bbox
        ox, oy, ow, oh = outer_bbox
        return (ix >= ox and iy >= oy and
                ix + iw <= ox + ow and iy + ih <= oy + oh)

    # Extract a region from the binary mask given a bounding box
    def extract_region_from_binary_mask(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        return self.binary_mask[y:y + h, x:x + w]

    # Static method to sort bounding boxes
    @staticmethod
    def sort_bboxes(bboxes):
        # If there are less than 2 bboxes, return as is
        if len(bboxes) < 2:
            return bboxes

        # Determine whether to sort by x or y coordinate
        first_bbox, second_bbox = bboxes[:2]
        sort_index = 0 if abs(second_bbox[0] - first_bbox[0]) > abs(second_bbox[1] - first_bbox[1]) else 1
        # Sort the bboxes based on the determined index
        return sorted(bboxes, key=lambda bbox: bbox[sort_index])

    # Static method to remove contained regions from a list of bounding boxes
    @staticmethod
    def remove_overlaping_regions(region_bboxes):
    # If there are less than 2 bboxes, return as is
        if len(region_bboxes) < 2:
            return region_bboxes

        def is_overlap(box1, box2):
            (ix, iy, iw, ih) = box1
            (jx, jy, jw, jh) = box2
            return (ix < jx + jw and ix + iw > jx and
                    iy < jy + jh and iy + ih > jy)

        def merge_boxes(box1, box2):
            (ix, iy, iw, ih) = box1
            (jx, jy, jw, jh) = box2
            new_x = min(ix, jx)
            new_y = min(iy, jy)
            new_w = max(ix + iw, jx + jw) - new_x
            new_h = max(iy + ih, jy + jh) - new_y
            return (new_x, new_y, new_w, new_h)

        combined_bboxes = list(region_bboxes)
        i = 0
        while i < len(combined_bboxes):
            j = i + 1
            merged = False
            while j < len(combined_bboxes):
                if is_overlap(combined_bboxes[i], combined_bboxes[j]):
                    combined_bboxes[i] = merge_boxes(combined_bboxes[i], combined_bboxes[j])
                    combined_bboxes.pop(j)
                    merged = True
                else:
                    j += 1
            if not merged:
                i += 1

        return combined_bboxes
