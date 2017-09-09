import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time, os


class model_RNN(object):
	def __init__(self, 
				 sess, 
				 batch_size=16, 
				 learning_rate=0.001,
				 num_layers = 2,
				 num_vocab = 1,
				 hidden_layer_units = 64,
				 sequence_length = 8,
				 data_dir='preprocessed_data/',
				 checkpoint_dir='checkpoint/', 
				 sample_dir=None):

		self.sess = sess
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.num_layers = num_layers
		self.hidden_layer_units = hidden_layer_units
		self.num_vocab = num_vocab
		self.sequence_length = sequence_length
		self.data_dir = data_dir
		self.checkpoint_dir = checkpoint_dir

		# input place holders
		self.X = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name='input')
		self.Y = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name='label')
		
		# self.X_gen = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input')
		# self.Y_gen = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input')
		
		self.x_one_hot = tf.one_hot(self.X, self.num_vocab)
		self.y_one_hot = tf.one_hot(self.Y, self.num_vocab)

		# self.x_gen_one_hot = tf.one_hot(self.X_gen, self.num_vocab)
		# self.y_gen_one_hot = tf.one_hot(self.Y_gen, self.num_vocab)

		self.optimizer, self.sequence_loss, self.curr_state = self.build_model()

		self.saver = tf.train.Saver()
		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


	def create_rnn_cell(self):
		cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.hidden_layer_units,
											state_is_tuple = True)
		return cell


	def create_rnn(self):
		
		multi_cells = tf.contrib.rnn.MultiRNNCell([self.create_rnn_cell()
												   for _ in range(self.num_layers)],
												   state_is_tuple=True)
		multi_cells = rnn.DropoutWrapper(multi_cells, input_keep_prob=0.9, output_keep_prob=0.9)

		# prepare initial state value
		self.rnn_initial_state = multi_cells.zero_state(self.batch_size, tf.float32)

		rnn_outputs, out_states = tf.nn.dynamic_rnn(multi_cells, self.x_one_hot, dtype=tf.float32, initial_state=self.rnn_initial_state)
		return rnn_outputs, out_states


	def build_model(self): 
		
		rnn_output, out_state = self.create_rnn()
		rnn_output_flat = tf.reshape(rnn_output, [-1, self.hidden_layer_units]) # [N x sequence_length, hidden]
		
		self.logits = tf.contrib.layers.fully_connected(rnn_output_flat, self.num_vocab, None)

		# for generation 
		y_softmax = tf.nn.softmax(self.logits)         # [N x seqlen, vocab_size]
		pred = tf.argmax(y_softmax, axis=1)       # [N x seqlen]
		self.pred = tf.reshape(pred, [self.batch_size, -1]) # [N, seqlen]



		y_flat = tf.reshape(self.y_one_hot, [-1, self.num_vocab]) # [N x sequence_length, vocab_size]

		losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_flat, logits=self.logits)
		sequence_loss = tf.reduce_mean(losses)

		opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(sequence_loss)

		tf.summary.scalar('training loss', sequence_loss)
		self.merged_summary = tf.summary.merge_all()
		
		return opt, sequence_loss, out_state


	## save current model
	def save_model(self, checkpoint_dir, step): 
		model_name = "melodyRNN.model"
		model_dir = "model"
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	## load saved model
	def load(self, checkpoint_dir):   
		print(" [*] Reading checkpoint...")

		model_dir = "model"
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def train(self, input_sequences, label_sequences, num_epochs): 

		## initialize                         
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		
		counter = 0
		start_time = time.time()

		## loading model 
		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		num_all_sequences = input_sequences.shape[0]
		num_batches = int(num_all_sequences / self.batch_size)

		loss_per_epoch = []
		## training loop
		for epoch in range(num_epochs):
			for batch_idx in range(num_batches):
				start_time = time.time()
				losses_per_epoch = []
			
				_, loss, logits, curr_state, summary_str = self.sess.run([self.optimizer, 
															 self.sequence_loss, 
															 self.logits, 
															 self.curr_state,
															 self.merged_summary], 
							  feed_dict={
								self.X: input_sequences[batch_idx * self.batch_size:(batch_idx+1)*self.batch_size], 
								self.Y: label_sequences[batch_idx * self.batch_size:(batch_idx+1)*self.batch_size] 
								})

				self.writer.add_summary(summary_str, epoch)

				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
					% (epoch, batch_idx, num_batches,
						time.time() - start_time,
						loss))
				losses_per_epoch.append(loss)
			loss_per_epoch.append(np.mean(np.array(losses_per_epoch)))
			
			counter += 1

			# if np.mod(counter, 10) == 1:
			self.save_model(self.checkpoint_dir, counter)

				# # Get sample 
				# if np.mod(counter, 200) == 1:
				# 	self.get_sample(epoch, idx, 'train')
				# 	self.get_sample( epoch, idx, 'val')

				# # Saving current model
				# if np.mod(counter, 500) == 2:
				# 	self.save(args.checkpoint_dir, counter)
			np.savetxt('avg_loss_txt/averaged_loss_per_epoch_' + str(epoch) + '.txt', loss_per_epoch) 


	## generate melody from input
	def predict(self, user_input_sequence, mel_i_v):
		self.predict_opt = True
		print("User input : ", user_input_sequence.shape)

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		## prepare input sequence
		print '[1] preparing user input data' # done at prdeict.py

		## generate corresponding melody
		print '[2] generating sequence from RNN'
		# get the hidden state / logit(to get the last one as input)
		print 'firstly, iterating through input'
		for i in range(user_input_sequence.shape[0]):
			print i
			print user_input_sequence[i]
			new_logits, prediction = self.sess.run([self.logits, self.pred], 
								  				feed_dict={self.X: user_input_sequence[i]})
			print new_logits
			print prediction 
		print(new_logits.shape)

		print 'secondly, generating'
		generated_input_seq = []

		for one_hot in new_logits:
			generated_input_seq.append(np.argmax(one_hot))
		generated_input_seq = np.expand_dims(np.array(generated_input_seq), axis=0)

	
		## generate melody 
		generated_melody = []
		generated_melody_length = 0

		while(generated_melody_length < 4):

			generated_pred = self.sess.run([self.pred], 
							  				feed_dict={self.X: generated_input_seq})
			for p in generated_pred[0][0]:
				curr_curve = mel_i_v[p]
				generated_melody_length += curr_curve[1]
				if generated_melody_length > 4:
					break
				else:
					generated_melody.append(curr_curve)
					generated_input_seq = generated_pred[-1]


		# generated_logits = []
		# predicate = lambda x: len(x) < self.sequence_length
		# while(predicate(generated_logits)):
		# 	generated_input_seq, curr_state = self.sess.run([self.logits, self.curr_state], 
		# 						  				feed_dict={self.X: generated_input_seq,
		# 						  						   self.rnn_initial_state: curr_state})
		# 	generated_logits.append(generated_input_seq[-1])

		print(np.array(generated_melody).shape)
		
		# generated_output_seq = []
		
		# for one_hot in generated_logits[0]:
		# 	generated_output_seq.append(np.argmax(one_hot))
		# generated_output_seq = np.expand_dims(np.array(generated_output_seq), axis=0)
		
		# return generated_output_seq

		return generated_melody




