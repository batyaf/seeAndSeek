from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import os
from CharacterRecognizer import CharacterRecognizer
from TextExtractor import TextExtractor
from TextImageProcessor import TextImageProcessor


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_DIR = 'uploads'
PROCESSED_DIR = 'processed'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
extractor = TextExtractor()
processor = TextImageProcessor(extractor)
item_list_set = False

def process(image_bytes, items_list):
    #if not items_list:
    new_image, similar_sentences, similar_bounding_boxes = processor.seek_words_in_image(image_bytes)
    return new_image, similar_sentences

@socketio.on('set_item_list')
def handle_set_list(data):
    global item_list_set
    if 'itemList' not in data:
        emit('error', {'message': 'Missing itemList in request'})
        return
    items_list = data['itemList']
    if not items_list:
        emit('error', {'message': 'no search words'})
        return null
    processor.set_list_of_sentences(items_list)
    item_list_set = True
    emit('items_set', {})


@socketio.on('process_frame')
def handle_process_frame(data):
    global item_list_set
    try:
        if 'imageData' not in data or 'itemList' not in data:
            emit('error', {'message': 'Missing imageData or itemList in request'})
            return
        # Wait for up to 100 milliseconds (10 checks * 10 milliseconds)
        for _ in range(10):
            if item_list_set:
                break
            time.sleep(0.01)  # 10 milliseconds

        if not item_list_set:
            emit('error', {'message': 'Item list not set. Please set the item list first.'})
            return

        image_data = data['imageData']
        item_list = data['itemList']
        image_bytes = base64.b64decode(image_data.split(',')[1])
        new_image, similar_sentences = process(image_bytes, item_list)
        if len(similar_sentences)>0:
            base64_image = base64.b64encode(new_image).decode('utf-8')
            emit('marked_frame', {
                'image': base64_image,
                'sentences': similar_sentences
            })






    except Exception as e:
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5050, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)