import simplejson as json
from flask import Flask
from flask import send_from_directory, request, jsonify
from flask_cors import CORS
import os

from web_server.common.melody import Melody
import time
import random

app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/js/<path:path>')
def send_js(path):
    print(path)
    return app.send_static_file(os.path.join('js', path))

@app.route('/images/<path:path>')
def send_images(path):
    return app.send_static_file(os.path.join('images', path))

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/duet', methods=['POST'])
def duet():
    '''

    :return:
        [
            {
                'pitch': 0~127숫자값,
                'duration': 0~elementsPerMeasure,
                'offset': 0~elementsPerMeasure,
                'velocity': 0~127숫자값
            }
        ]
    '''
    now = time.time()
    input_melody = json.loads(request.data)

    # char_rnn_melody = Melody.createCharRNNSequence(input_melody)
    char_rnn_melody = Melody.createCharGenerationSequence(input_melody)

    return jsonify(char_rnn_melody)
    # return jsonify(dummy_melody)

if __name__ == '__main__':
    app.run(debug=True)
