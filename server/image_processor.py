import io
from google.cloud import vision
import Levenshtein as lev
from PIL import Image, ImageDraw

def get_image_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    # Convert the image to bytes for Google Vision API
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()



def extract_text_from_image(image_bytes):
    client = vision.ImageAnnotatorClient.from_service_account_json(r"C:\see_and_seek\clientfile_vision_ai.json")

    #with io.open(image_path, 'rb') as image_file:
    #    content = image_file.read()
    content = get_image_from_bytes(image_bytes)
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    extracted_sentences = []
    bounding_boxes = []
    for text in texts[1:]:  # Skip the first item which is the full text
        extracted_sentences.append(text.description)
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        bounding_boxes.append(vertices)

    return extracted_sentences, bounding_boxes

def find_similar_sentences(extracted_sentences, reference_sentences, threshold=0.8):
    similar_sentences = []
    for extracted_sentence in extracted_sentences:
        for reference_sentence in reference_sentences:
            distance = lev.distance(extracted_sentence.lower(), reference_sentence.lower())
            max_length = max(len(extracted_sentence), len(reference_sentence))
            similarity = 1 - (distance / max_length)
            if similarity >= threshold:
                similar_sentences.append((extracted_sentence, reference_sentence, similarity))
    return similar_sentences


def draw_bounding_boxes(self, image_bytes, bounding_boxes):
    image = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    for box in bounding_boxes:
        draw.polygon(box, outline='red', width=5)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

def same_sentences(image_path, reference_sentences):
    extracted_sentences, bounding_boxes = extract_text_from_image(image_path)
    similar_sentences = find_similar_sentences(extracted_sentences, reference_sentences)
    if similar_sentences:
        for extracted, reference, similarity in similar_sentences:
            print(f"Extracted: {extracted}, Reference: {reference}, Similarity: {similarity:.2f}")

        # Find the bounding boxes for the similar sentences
        similar_bounding_boxes = []
        for extracted, _, _ in similar_sentences:
            for i, sentence in enumerate(extracted_sentences):
                if sentence == extracted:
                    similar_bounding_boxes.append(bounding_boxes[i])
                    break

        # Draw bounding boxes on the image
        image_with_boxes = draw_bounding_boxes(image_path, similar_bounding_boxes)
        return image_with_boxes