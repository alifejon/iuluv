import music21

o = music21.converter.parse('TheDanceOfEternity_dr.mid')

dr_stream = music21.stream.Stream()
drum_data = []

for n in o.parts[0].getElementsByClass(music21.chord.Chord):
    for p in n.pitches:
        drum_data.append([p.midi, n.offset])
        
for n in o.parts[0].getElementsByClass(music21.note.Note):
    drum_data.append([n.pitch.midi, n.offset])
    
    
import numpy as np

drum_data = np.array(drum_data)
drum_data = drum_data[drum_data[:, 1].argsort()]

drum_data_diff = []
drum_data_diff.append([drum_data[0,0], 0])

for n in range(0, len(drum_data) - 2):
    drum_data_diff.append([drum_data[n+1, 0], drum_data[n+1,1] - drum_data[n,1]])
    
drum_data_diff = np.array(drum_data_diff)
drum_note_set = set(drum_data_diff[:,0])
drum_timing_diff_set = set(drum_data_diff[:,1])

drum_data_diff_1d = []
for n in range(0, len(drum_data_diff) - 1):
    drum_data_diff_1d.append(drum_data_diff[n,0] + drum_data_diff[n,1] * 1j)
    
drum_note_set = list(set(drum_data_diff_1d))
drum_note_dic = {w: i for i, w in enumerate(drum_data_diff_1d)}

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility

data_dim = len(drum_note_set)
hidden_size = len(drum_note_set)
num_classes = len(drum_note_set)
sequence_length = 100  # Any arbitrary number
learning_rate = 0.1

