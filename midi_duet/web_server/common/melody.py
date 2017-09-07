import random

import tensorflow as tf
import os
from six.moves import cPickle
from common.model import Model
from six import text_type

save_dir = 'save'
sample = 10
n = 10
prime = 'GGAAGGEGG'

with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
    saved_args = cPickle.load(f)
with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
    chars, vocab = cPickle.load(f)
model = Model(saved_args, training=False)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(model.sample(sess, chars, vocab, n, prime, sample).encode('utf-8'))

noteList = [60, 62, 64, 65, 67, 69, 71, 72]
chordList = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'H']

noteToChord = dict(zip(noteList, chordList))
chordToNote = dict(zip(chordList, noteList))

class Melody:
    def __init__(self, pitch=0, duration=0, offset=0, velocity=0):
        '''
        'pitch': 0~127숫자값,
        'duration': 0~elementsPerMeasure,
        'offset': 0~elementsPerMeasure,
        'velocity': 0~127숫자값
        '''

        self.pitch = pitch
        self.duration = duration
        self.offset = offset
        self.velocity = velocity
        pass

    @classmethod
    def createRandom(cls, measureInSec=None, elementsPerMeasure=None):
        if measureInSec is None:
            measureInSec = 5
        if elementsPerMeasure is None:
            elementsPerMeasure = 32

        pitch = random.randrange(0, 128)
        duration = random.randrange(0, elementsPerMeasure)
        offset = random.randrange(0, elementsPerMeasure)
        velocity = random.randrange(0, 128)
        melody = Melody(pitch=pitch, duration=duration, offset=offset, velocity=velocity)

        return melody

    @classmethod
    def createCharRNN(cls, input_melody, measureInSec=None, elementsPerMeasure=None):
        pitch = random.randrange(0, 5)
        duration = random.randrange(0, elementsPerMeasure)
        offset = random.randrange(0, elementsPerMeasure)
        velocity = random.randrange(0, 128)
        melody = Melody(pitch=pitch, duration=duration, offset=offset, velocity=velocity)

        return melody

    def toJSON(self):
        return {
            'pitch': self.pitch,
            'duration': self.duration,
            'offset': self.offset,
            'velocity': self.velocity,
        }