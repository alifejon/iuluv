import music21
import numpy as np


def truncate(f, n):
	'''Truncates/pads a float f to n decimal places without rounding'''
	s = '%.12f' % f
	i, p, d = s.partition('.')
	return float('.'.join([i, (d+'0'*n)[:n]]))


def create_mel_data_each_file(midi_obj):
	
	mel_data = dict()

	print('# of notes : {}'.format(len(midi_obj.flat.getElementsByClass(music21.chord.Chord))))
	for n in midi_obj.flat.getElementsByClass(music21.chord.Chord):
		# print(n)
		n_offset = truncate(n.offset, 6)
		# print(n_offset)
		# print('----')
		# print("chord {} : {}".format(n, n_offset))
		# print("notes {}".format(n.pitches))
		if n_offset not in mel_data:
			mel_data[n_offset] = dict()
		mel_data[n_offset]['offset'] = n_offset
		for p in n.pitches:
			# print(p)
			mel_data[n_offset]['note'] = p.midi
			# print("note {} : {}: ".format(p.midi, n_offset))

	print('# of chords : {}'.format(len(midi_obj.flat.getElementsByClass(music21.note.Note))))
	for n in midi_obj.flat.getElementsByClass(music21.note.Note):
		n_offset = truncate(n.offset, 6)
		# print('----')
		# print("note {} : offset {}".format(n, n_offset))
		# print("note pitch {}".format(n.pitches))
		if n_offset not in mel_data:
			mel_data[n_offset] = dict()
		mel_data[n_offset]['offset'] = n_offset
		prev_p = 0
		for p in n.pitches:
			if prev_p < p.midi:
				mel_data[n_offset]['note'] = p.midi
			prev_p = p.midi    
			# print("note {} : {}: ".format(p.midi, n_offset))  
	
	print('mel_data : ', len(list(mel_data)))
	return mel_data # note pitch / note offset for each file

def create_curve_seq(mel_arr):
	curve_seq = []
	for idx in range(1, len(mel_arr)):
		curr_p_diff = mel_arr[idx]['note'] - mel_arr[idx-1]['note']
		curr_t_diff = truncate(mel_arr[idx]['offset'] - mel_arr[idx-1]['offset'], 5)
		curve_seq.append((curr_p_diff, curr_t_diff))
	# curve_seq = []
	# for idx in range(1, len(mel_arr)):
	#     curr_p_diff = mel_arr[idx]['note'] - mel_arr[idx-1]['note']
	#     curr_t_diff = mel_arr[idx]['offset'] - mel_arr[idx-1]['offset']
	#     curve_seq.append((curr_p_diff, curr_t_diff))

	return curve_seq


def euclidean_dist(a, b):
	return np.linalg.norm(np.array(a)-np.array(b))

def find_similar_curve(query_curve, mel_set):
	list_mel_set = list(mel_set)
	min_dist = 10000 # to act like positive infinity
	found_curve_idx = -1 # just to initialize
	for idx, curve in enumerate(list_mel_set):
#         print(curve)
		if np.array_equal(query_curve, curve):
			found_curve_idx = idx
			break
		elif euclidean_dist(query_curve, curve) < min_dist:
			min_dist = euclidean_dist(query_curve, curve)
			found_curve_idx = idx
	print(list_mel_set[found_curve_idx])
	return found_curve_idx
