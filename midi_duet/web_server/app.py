import simplejson as json
from flask import Flask
from flask import send_file, request, jsonify

from web_server.common.melody import Melody
import time

# app = Flask(__name__, static_url_path='', static_folder=os.path.abspath('../static'))
app = Flask(__name__)

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
