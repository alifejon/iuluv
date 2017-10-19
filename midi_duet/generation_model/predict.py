#-*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import pickle
import numpy as np

from generation_model.model import model_RNN
from generation_model.mel_op import *


parser = argparse.ArgumentParser(description='')

parser.add_argument('--data_dir', dest='data_dir', default='./preprocessed_data/', help='data path')
parser.add_argument('--model_dir', dest='model_dir', default='./model/', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample/', help='sample are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint/', help='checkpoints are saved here')

args = parser.parse_args()

## prepare as an batch
def get_input_batch_sequence(input_seq, sequence_length):
    
    input_sequence_batches = []

    num_seqs_per_song = max(int((len(input_seq) / sequence_length)) - 1, 0)

    for ns in range(num_seqs_per_song):
        batch = np.expand_dims(input_seq[ns * sequence_length:(ns+1) * sequence_length], axis=0)
        input_sequence_batches.append(batch)

    return np.array(input_sequence_batches)


def preprocess_user_input(mel_arr):
    curve_seq_list = []
    curve_seq_list.append(create_curve_seq(mel_arr))
    print(len(curve_seq_list))

    return curve_seq_list


def predict_output(curve_arr, sequence_length = 8):

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

    with tf.Session() as sess:
        model = model_RNN(sess, 
                         batch_size=1, 
                         learning_rate=0.001,
                         num_layers = 3,
                         num_vocab = vocab_size,
                         hidden_layer_units = 64,
                         sequence_length = 8,
                         data_dir='generation_model/preprocessed_data/')

    output_sequence = model.predict(np.array(input_sequence_as_batches), mel_i_v)

    return output_sequence


if __name__ == "__main__":

    sequence_length = 8

    data_dir = args.data_dir
    print(data_dir)
    # data = []
    with open(data_dir + "mel_data.p", "rb") as fp:   
        data = pickle.load(fp)
    mel_set, mel_v_i, mel_i_v, vocab_size = (data[i] for i in range(len(data)))

    print('length of mel_set', len(mel_set))

    with open(data_dir + "mel_arr_list.p", "rb") as fp:   
        mel_arr_list = pickle.load(fp)

    with open(data_dir + "curve_arr_list.p", "rb") as fp:   
        curve_arr_list = pickle.load(fp)

    user_input_file = 'data/midi_gt_solo/G_Scarred.mid'
    midi_obj = music21.converter.parse(user_input_file)
    mel_data = create_mel_data_each_file(midi_obj)
    mel_arr = []
    for key in sorted(mel_data.keys()):
        mel_arr.append(mel_data[key])
    
    curve_arr = create_curve_seq(mel_arr)
    output_sequence = predict_output(curve_arr, sequence_length)

    print(output_sequence)

