import random

import tensorflow as tf
import os
from six.moves import cPickle
from web_server.common.model import Model
from six import text_type
import math

import pickle
from generation_model.model import model_RNN
from generation_model.mel_op import *

save_dir = 'web_server/save'
sample = 10
n = 10
prime = 'GGAAGGEGG'

# with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
#     saved_args = cPickle.load(f)
# with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
#     chars, vocab = cPickle.load(f)
# model = Model(saved_args, training=False)
#
# sess = tf.Session()
# sess.as_default()
# tf.global_variables_initializer().run(session=sess)
# saver = tf.train.Saver(tf.global_variables())
# ckpt = tf.train.get_checkpoint_state(save_dir)
# if ckpt and ckpt.model_checkpoint_path:
#     saver.restore(sess, ckpt.model_checkpoint_path)
#
# print(model.sample(sess, chars, vocab, n, prime, sample))
# print(model.sample(sess, chars, vocab, n, 'CECECECECE', sample))

noteList = [60, 62, 64, 65, 67, 69, 71, 72]
chordList = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'H']

noteToChord = dict(zip(noteList, chordList))
chordToNote = dict(zip(chordList, noteList))


sequence_length = 8
data_dir = 'generation_model/preprocessed_data/'
print(data_dir)
# data = []
with open(data_dir + "mel_data.p", "rb") as fp:
    data = pickle.load(fp)
mel_set, mel_v_i, mel_i_v, vocab_size = (data[i] for i in range(len(data)))
with tf.Session() as sess:
    model = model_RNN(sess,
          batch_size=1,
          learning_rate=0.001,
          num_layers=3,
          num_vocab=vocab_size,
          hidden_layer_units=64,
          sequence_length=8,
          data_dir='generation_mode/preprocessed_data/')

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
    def createCharGenerationSequence(cls, input_melody, measureInSec=None, elementsPerMeasure=None):
        # add note vlaue
        for x in input_melody:
            x['note'] = x['pitch']






        ## prepare as an batch
        def get_input_batch_sequence(input_seq, sequence_length):

            input_sequence_batches = []

            num_seqs_per_song = max(int((len(input_seq) / sequence_length)) - 1, 1)

            for ns in range(num_seqs_per_song):
                batch = np.expand_dims(input_seq[ns * sequence_length:(ns + 1) * sequence_length], axis=0)
                input_sequence_batches.append(batch)

            return np.array(input_sequence_batches)

        def predict_output(curve_arr, sequence_length=8):

            ## prepare user input sequence with existing vocab in melody set
            user_input_sequence = []
            for curve in curve_arr:
                similar_curve = find_similar_curve(curve, mel_set)
                user_input_sequence.append(similar_curve)

            print(user_input_sequence)

            ## pad zeros to the user input sequence
            if len(user_input_sequence) < sequence_length:
                user_input_sequence += [0] * (sequence_length - len(user_input_sequence))

            input_sequence_as_batches = get_input_batch_sequence(user_input_sequence, sequence_length)



            output_sequence = model.predict(np.array(input_sequence_as_batches), mel_i_v)

            return output_sequence


        print(input_melody)
        curve_arr = create_curve_seq(input_melody)
        output_sequence = predict_output(curve_arr, sequence_length)

        return input_melody

    @classmethod
    def createCharRNNSequence(cls, input_melody, measureInSec=None, elementsPerMeasure=None):
        print(input_melody)
        charSequence = [noteToChord[mel['pitch']] for mel in input_melody]
        charSequenceInput = ''.join(charSequence)


        if len(charSequenceInput) < sample:
            charSequence = charSequenceInput * math.ceil(sample / len(charSequenceInput))
            charSequence = charSequence[:sample]

        tf.reset_default_graph()
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

            newCharSeq = model.sample(sess, chars, vocab, n, charSequence, sample)
            print('newCharSeq', newCharSeq, newCharSeq[len(charSequence):])
            newCharSeq = newCharSeq[len(charSequence):]
            newCharSeq = newCharSeq[:len(charSequenceInput)]
            print(charSequence, newCharSeq)

            newPitchs = [chordToNote[char] for char in newCharSeq]
            for idx, pitch in enumerate(newPitchs):
                input_melody[idx]['pitch'] = pitch

        return input_melody

    def toJSON(self):
        return {
            'pitch': self.pitch,
            'duration': self.duration,
            'offset': self.offset,
            'velocity': self.velocity,
        }