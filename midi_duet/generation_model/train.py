#-*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import pickle
import numpy as np

from generation_model.model import model_RNN

parser = argparse.ArgumentParser(description='')

parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=2, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# of data in one batch')
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='sequence length')
parser.add_argument('--sequence', dest='sequence_length', type=int, default=8, help='sequence length')

parser.add_argument('--data_dir', dest='data_dir', default='./preprocessed_data/', help='data path')
parser.add_argument('--model_dir', dest='model_dir', default='./model/', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample/', help='sample are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint/', help='checkpoints are saved here')

args = parser.parse_args()


if __name__ == "__main__":

    data_dir = args.data_dir
    print(data_dir)

    sequence_length = args.sequence_length
    # data = []
    with open(data_dir + "mel_data.p", "rb") as fp:   
        data = pickle.load(fp)
    mel_set, mel_v_i, mel_i_v, vocab_size = (data[i] for i in range(len(data)))

    with open(data_dir + "mel_arr_list.p", "rb") as fp:   
        mel_arr_list = pickle.load(fp)

    with open(data_dir + "curve_arr_list.p", "rb") as fp:   
        curve_arr_list = pickle.load(fp)

    print("number of songs : ", len(mel_arr_list))

    def get_mel_id_sequence(mel_arr_list, curve_arr_list):
        
        input_seq_list = []
        label_seq_list = []

        for curve_arr in curve_arr_list:
            print(len(curve_arr))
            mel_id_seq = []
            for curve in curve_arr:
                mel_id = mel_v_i[curve]
                mel_id_seq.append(mel_id)

            input_id_seq = np.array(mel_id_seq)[:-sequence_length]
            label_id_seq = np.roll(np.array(mel_id_seq), -sequence_length)[:-sequence_length]
            # label_id_seq = np.array(mel_id_seq)[:-sequence_length]
            print('input_id_seq', input_id_seq)
            print('label_id_seq', label_id_seq)

            input_seq_list.append(input_id_seq)
            label_seq_list.append(label_id_seq)

        return input_seq_list, label_seq_list

    input_seq_list, label_seq_list = get_mel_id_sequence(mel_arr_list, curve_arr_list)



    def get_batch_sequence(input_seq_list, label_seq_list, sequence_length):
        
        input_sequences = []
        label_sequences = []

        for idx in range(len(input_seq_list)):
            num_seqs_per_song = max(int((len(input_seq_list[idx]) / sequence_length)) - 1, 0)

            for ns in range(num_seqs_per_song):
                input_sequences.append(input_seq_list[idx][ns * sequence_length:(ns+1) * sequence_length])
                label_sequences.append(label_seq_list[idx][ns * sequence_length:(ns+1) * sequence_length])

        return np.array(input_sequences), np.array(label_sequences)

    input_sequences, label_sequences = get_batch_sequence(input_seq_list, label_seq_list, args.sequence_length)


    with tf.Session() as sess:
        model = model_RNN(sess, 
                         batch_size=16, 
                         learning_rate=0.001,
                         num_layers = 3,
                         num_vocab = vocab_size,
                         hidden_layer_units = 64,
                         sequence_length = 8,
                         data_dir='preprocessed_data/')

        model.train(input_sequences, label_sequences, args.num_epochs)

