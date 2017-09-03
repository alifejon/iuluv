import simplejson as json
from flask import Flask
from flask import send_from_directory, request, jsonify
from flask_cors import CORS
import os

from common.melody import Melody
import time

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
    print(request.data)
    input_melody = json.loads(request.data)

    print('request.data', input_melody)

    dummy_melody = [Melody.createRandom().toJSON() for _ in range(len(input_melody))]
    dummy_melody = sorted(dummy_melody, key=lambda k: k['offset'])
    print('dummy_melody', dummy_melody)

    return jsonify(dummy_melody)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return send_file('../static/index.html')

if __name__ == '__main__':
    app.run(debug=True)
