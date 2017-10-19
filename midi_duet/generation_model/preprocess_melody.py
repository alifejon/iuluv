import music21
import os
import pickle
import argparse

from generation_model.mel_op import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--loadpre', type=bool, default=False, help='do you just wanna load txt?')
    parser.add_argument('--preprocessed_dir', type=str, default='./preprocessed_data/', help='dir to save processed data.')

    args = parser.parse_args()
    data_dir = args.preprocessed_dir

    if args.loadpre == False:

        file_list = []
        data_path = './data/midi_gt_solo/'

        for file_name in os.listdir(data_path):
            if file_name.endswith('.mid') or file_name.endswith('.midi'):
                file_list.append(data_path + file_name)

        mel_arr_list = []

        for file_name in file_list:
            print(file_name)
            midi_obj = music21.converter.parse(file_name)
            mel_data = create_mel_data_each_file(midi_obj) # note pitch / note offset for each file

            mel_arr = []
            # key : n_offset -> to be sorted
            for key in sorted(mel_data.iterkeys()):
                mel_arr.append(mel_data[key])
            
            mel_arr_list.append(mel_arr)
            

        with open(data_dir + "mel_arr_list.p", "wb") as fp:   #Pickling
            pickle.dump(mel_arr_list, fp)

    else:
        with open(data_dir + "mel_arr_list.p", "rb") as fp:   # Unpickling
            mel_arr_list = pickle.load(fp)


    curve_seq_list = []
    for mel_arr in mel_arr_list:
        curve_seq_list.append(create_curve_seq(mel_arr))

    print(len(curve_seq_list))

    with open(data_dir + "curve_arr_list.p", "wb") as fp:   #Pickling
        pickle.dump(curve_seq_list, fp)


    ## clustering phase to give id
    ## (using spectral clustering)

    '''
    NOT applied.

    '''

    # flatten
    curve_corpus= sum(curve_seq_list, [])
    print(len(curve_corpus))
    
    def get_corpus_data(curve_corpus):
        curve_corpus_set = set(curve_corpus) 
        val_indices = dict((v, i) for i, v in enumerate(curve_corpus_set))
        indices_val = dict((i, v) for i, v in enumerate(curve_corpus_set))

        return curve_corpus_set, val_indices, indices_val

    mel_set, mel_v_i, mel_i_v = get_corpus_data(curve_corpus)
    vocab_size = len(mel_set)
    print(vocab_size)

    data = [mel_set, mel_v_i, mel_i_v, vocab_size]

    with open(data_dir + "mel_data.p", "wb") as fp:   #Pickling
        pickle.dump(data, fp)


