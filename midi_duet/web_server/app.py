import simplejson as json
from flask import Flask
from flask import send_from_directory, request, jsonify
from flask_cors import CORS
import os

from web_server.common.melody import Melody
import time
import random

from flask_socketio import SocketIO, emit

app = Flask(__name__, static_url_path='/static')
app.secret_key = "secret"
# CORS(app)
socketio = SocketIO(app)

user_no = 1


@app.route('/img/<path:path>')
def send_img(path):
    print(path)
    return app.send_static_file(os.path.join('img', path))

@app.route('/sounds/<path:path>')
def send_sounds(path):
    print(path)
    return app.send_static_file(os.path.join('sounds', path))

@app.route('/styles/<path:path>')
def send_styles(path):
    print(path)
    return app.send_static_file(os.path.join('styles', path))

@app.route('/webcomponents/<path:path>')
def send_webcomponents(path):
    print(path)
    return app.send_static_file(os.path.join('webcomponents', path))

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

@app.route('/recv_test')
def recv_test():
    return app.send_static_file('receiver.html')

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

@app.route('/event', methods=['POST'])
def event():
    input_melody = json.loads(request.data)
    emit('started', input_melody, broadcast=True, namespace='/visual')

@socketio.on('connect', namespace='/visual')
def connect():
    print('connected!')

@socketio.on('start', namespace='/visual')
def connect(d):
    print('start', d);
    emit('started', d, broadcast=True)

@socketio.on('disconnect', namespace='/visual')
def disconnect():
    print("Disconnected")

@socketio.on('music_signal', namespace='/visual')
def receiveSignal(signal):
    print('music_signal', signal)
    emit('music_signal', signal, broadcast=True)

if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0')
    socketio.run(app, debug=True, host='0.0.0.0')
