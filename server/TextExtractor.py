import io
import os
from google.cloud import vision
import Levenshtein as lev
from PIL import Image, ImageDraw
import cv2
from CharacterRecognizer import CharacterRecognizer
from ImageTextExtractor import ImageTextExtractor
from structs import WordInfo
import numpy as np
from typing import List, Tuple
import random
import math
import time
import statistics




class TextExtractor:
    def __init__(self):
        characters = ['א', 'ב', 'ח', 'ד', 'ע', 'ג', 'ה', 'ך', 'כ', 'ק', 'ל', 'ם', 'מ', 'ן', 'נ', 'ף', 'פ', 'ר', 'ס', 'ש',
                      'ת', 'ט', 'ו', 'י', 'צ', 'ץ', 'ז']
        model_path = r"C:\SeeNSeek\server\hebrew_m.h5"
        self.char_recognizer = CharacterRecognizer(model_path, characters)
        print("loaded recognizer")



    def get_words_info(self, word_info_array: List[WordInfo]) -> Tuple[
        List[str], List[Tuple[int, int, int, int]]]:
        predicted_words = []
        bounding_boxes = []

        for word_info in word_info_array:
            # Get the bounding box
            bounding_box = word_info.bounding_box
            # Join the characters in reverse order to form the word
            predicted_word = ''.join(region.predicted_char for region in reversed(word_info.regions))
            # Append to respective lists
            predicted_words.append(predicted_word)
            bounding_boxes.append(bounding_box)

        return predicted_words, bounding_boxes

    def extract_text_from_image(self, image_bytes):
        extractor = ImageTextExtractor(image_bytes, self.char_recognizer, output_dir="extracted_regions")
        detected_words = extractor.process()
        extracted_sentences,  bounding_boxes = self.get_words_info(detected_words)
        return extracted_sentences, bounding_boxes


