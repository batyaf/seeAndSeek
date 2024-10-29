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

class TextImageProcessor:
    # # Define weight_matrix as a class attribute
    # weight_matrix = {
    #     ('ש', 'ע'): 0.25, ('ע', 'ש'): 0.25,
    #     ('נ', 'כ'): 0.25, ('כ', 'נ'): 0.25,
    #     ('ל', 'ת'): 0.40, ('ת', 'ל'): 0.40,
    #     ('י', 'ו'): 0.10, ('ו', 'י'): 0.10,
    #     ('ן', 'ו'): 0.10, ('ו', 'ן'): 0.10,
    #     ('י', 'ן'): 0.10, ('ן', 'י'): 0.10,
    #     ('ך', 'ן'): 0.10, ('ן', 'ך'): 0.10,
    #     ('י', 'ר'): 0.30, ('ר', 'י'): 0.30,
    #     ('ך', 'ר'): 0.20, ('ר', 'ך'): 0.20,
    #     ('ק', 'ה'): 0.15, ('ה', 'ק'): 0.15,
    #     ('ר', 'ד'): 0.30, ('ד', 'ר'): 0.30,
    #     ('י', 'ם'): 0.20, ('ס', 'י'): 0.20,
    #     # Add other weights for other visually similar characters
    # }

    def __init__(self,extractor):
        self.list_of_sentences = []
        self.text_extractor = extractor
        self.mutations_dictionary = {}
        self.found_words = []

    def set_list_of_sentences(self, list_of_sentences):
        self.mutations_dictionary = {}
        self.list_of_sentences = list_of_sentences
        self.mutations_dictionary = self.create_mutation_dictionary(list_of_sentences, self.hebrew_interchangeable, self.priority_chars)

    def extract_text_from_image(self, image_bytes):
        return self.text_extractor.extract_text_from_image(image_bytes)

    def check_mutations(self, words_to_check):
        found_mutations = []
        not_found_words = []

        for i, word in enumerate(words_to_check):
            if word in self.mutations_dictionary:
                print(f"found {i} {word}")
                original, changes = self.mutations_dictionary[word]
                found_mutations.append((i, word, original, changes))
            else:
                print(f"not found {i} {word}")
                not_found_words.append((i, word))

        return found_mutations, not_found_words



    def seek_words_in_image(self, image_bytes):
        extracted_sentences, bounding_boxes = self.text_extractor.extract_text_from_image(image_bytes)
        if len(extracted_sentences) == 0:
            return image_bytes, [], []

        [print(word) for word in extracted_sentences]
        found_mutations, not_found_mutations = self.check_mutations(extracted_sentences)
        similar_sentences = self.find_similar_sentences(not_found_mutations)
        similar_sentences.extend(found_mutations)
        if len(similar_sentences) == 0:
            return image_bytes, [], []
        self.list_of_sentences = [s for s in self.list_of_sentences if s not in set(item[2] for item in similar_sentences)]
        # Get the bounding boxes for the similar sentences using the returned indices
        similar_bounding_boxes = [bounding_boxes[index] for index, _, _, _ in similar_sentences]
        new_image = self.draw_bounding_boxes(image_bytes, similar_bounding_boxes)
        return new_image, similar_sentences, similar_bounding_boxes

    def create_mutations(self,word, interchangeable_dict, priority_chars):
        mutations = {word: (word, 0)}  # Start with the original word and 0 changes
        word_length = len(word)

        priority_positions = [i for i, char in enumerate(word) if char in priority_chars]
        num_additional_chars = max(0, math.ceil(word_length / 3) - len(priority_positions))
        available_positions = [i for i in range(word_length) if i not in priority_positions]
        additional_positions = random.sample(available_positions, min(num_additional_chars, len(available_positions)))
        chars_to_replace = sorted(priority_positions + additional_positions)

        for pos in chars_to_replace:
            char = word[pos]
            if char in interchangeable_dict:
                replacements = [r for r in interchangeable_dict[char] if r != char]
                if replacements:
                    new_mutations = {}
                    for mutation, (original, changes) in mutations.items():
                        for replacement in replacements:
                            new_word = mutation[:pos] + replacement + mutation[pos + 1:]
                            new_mutations[new_word] = (original, changes + 1)
                    mutations.update(new_mutations)

        return mutations

    def create_mutation_dictionary(self, word_list, interchangeable_dict, priority_chars):
        mutation_dict = {}
        for word in word_list:
            word_mutations = self.create_mutations(word, interchangeable_dict, priority_chars)
            mutation_dict.update(word_mutations)
        return mutation_dict

    # Hebrew interchangeable characters dictionary
    hebrew_interchangeable = {
        'א': ['א'],
        'ב': ['ב', 'כ'],
        'ג': ['ג', 'נ'],
        'ד': ['ד', 'ר'],
        'ה': ['ה', 'ק', 'ר'],
        'ו': ['ו', 'י', 'ן'],
        'ז': ['ז'],
        'ח': ['ח', 'ת'],
        'ט': ['ט', 'פ'],
        'י': ['י', 'ו', 'ן'],
        'כ': ['כ', 'נ', 'ם'],
        'ל': ['ל','ת'],
        'מ': ['מ'],
        'נ': ['נ', 'ג'],
        'ס': ['ס' 'י', 'כ'],
        'ע': ['ע', 'ש'],
        'פ': ['פ', 'ט'],
        'צ': ['צ'],
        'ק': ['ק', 'ה', 'ר'],
        'ר': ['ר', 'ד', 'ך'],
        'ש': ['ש', 'ע'],
        'ת': ['ת', 'ל'],
        'ם': ['ם', 'י', 'כ'],
        'ן': ['ן', 'ו', 'י', 'ך'],
        'ך': ['ך', 'ר', 'ן'],
        'ץ': ['ץ'],
        'ף': ['ף'],
    }

    priority_chars = ['ן', 'ו', 'י', 'ר', 'ך', 'ם']

    # def weighted_levenshtein(self,s1, s2, weight_matrix):
    #     rows = len(s1) + 1
    #     cols = len(s2) + 1
    #     dist = [[0 for _ in range(cols)] for _ in range(rows)]
    #
    #     for i in range(rows):
    #         dist[i][0] = i
    #     for j in range(cols):
    #         dist[0][j] = j
    #
    #     for i in range(1, rows):
    #         for j in range(1, cols):
    #             if s1[i - 1] == s2[j - 1]:
    #                 cost = 0
    #             else:
    #                 cost = weight_matrix.get((s1[i - 1], s2[j - 1]), 1)
    #
    #             dist[i][j] = min(dist[i - 1][j] + 1,  # deletion
    #                              dist[i][j - 1] + 1,  # insertion
    #                              dist[i - 1][j - 1] + cost)  # substitution
    #
    #     return dist[-1][-1]
    #
    # # Super simple weighted matrix example
    #
    #
    # def find_similar_sentences(self,extracted_sentences, weight_matrix, threshold=0.8):
    #     similar_sentences = []
    #     for i, extracted_sentence in enumerate(extracted_sentences):
    #         for reference_sentence in self.list_of_sentences:
    #             distance = self.weighted_levenshtein(extracted_sentence, reference_sentence, weight_matrix)
    #             max_length = max(len(extracted_sentence), len(reference_sentence))
    #             similarity = 1 - (distance / max_length)
    #             if similarity >= threshold:
    #                 similar_sentences.append((i, extracted_sentence, reference_sentence, similarity))
    #     return similar_sentences

    def find_similar_sentences(self, extracted_sentences, threshold=0.8):
        similar_sentences = []
        for i, extracted_sentence in extracted_sentences:
            for reference_sentence in self.list_of_sentences:
                distance = lev.distance(extracted_sentence.lower(), reference_sentence.lower())
                max_length = max(len(extracted_sentence), len(reference_sentence))
                similarity = 1 - (distance / max_length)
                if similarity >= threshold:
                    similar_sentences.append((i, extracted_sentence, reference_sentence, similarity))
        return similar_sentences

    def draw_bounding_boxes(self, image_bytes, bounding_boxes):
        image = Image.open(io.BytesIO(image_bytes))
        image = image.rotate(-90, expand=True)
        draw = ImageDraw.Draw(image)
        for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box)
            # Draw rectangle
            draw.rectangle([x1, y1, x2 + x1, y2 + y1], outline='red', width=8)
            # Draw inner rectangle in blue
            draw.rectangle([x1 + 5, y1 + 5, x2 + x1 - 5, y2 + y1 - 5], outline='green', width=8)

        image = image.rotate(90, expand=True)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        return buffered.getvalue()

    def save_image_to_file(self, image_bytes, file_name):
        # Define the base path
        base_path = r"C:\proj\server\p_images"

        # Construct the full path
        full_path = os.path.join(base_path, file_name)

        # Ensure the directory exists
        os.makedirs(base_path, exist_ok=True)

        # Save the image bytes to the file
        with open(full_path, 'wb') as file:
            file.write(image_bytes)

    def load_image_from_file(self, file_name):
        base_path = r"C:\proj\server\s_images"
        # Construct the full path to the image
        image_path = os.path.join(base_path, file_name)

        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None

        try:
            # Open the image using Pillow
            with Image.open(image_path) as img:
                # Create a BytesIO buffer
                buffered = io.BytesIO()
                # Save the image to the buffer in PNG format
                img.save(buffered, format="PNG")
                # Get the image bytes
                image_bytes = buffered.getvalue()
                return image_bytes

        except Exception as e:
            print(f"Error: Unable to load image from {image_path}. Exception: {e}")
            return None
