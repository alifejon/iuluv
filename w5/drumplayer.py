import music21
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.set_random_seed(777)  # reproducibility


o = music21.converter.parse('TheDanceOfEternity_dr.mid')

dr_stream = music21.stream.Stream()
drum_data = []

for n in o.parts[0].getElementsByClass(music21.chord.Chord):
    #print('----')
    #print("chord {} : {}".format(n, n.offset))
    #print("notes {}".format(n.pitches))
    for p in n.pitches:
        #print("note {} : {}".format(p.midi, n.offset))
        drum_data.append([p.midi, n.offset])

for n in o.parts[0].getElementsByClass(music21.note.Note):
    #print("note {} : {}".format(n.pitch.midi, n.offset))
    drum_data.append([n.pitch.midi, n.offset])

#print(len(drum_data))

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

set(drum_data_diff_1d)
len(set(drum_data_diff_1d))

drum_note_set = list(set(drum_data_diff_1d))

drum_note_dic = {w: i for i, w in enumerate(drum_data_diff_1d)}


data_dim = len(drum_note_set)
hidden_size = len(drum_note_set)
num_classes = len(drum_note_set)
sequence_length = 100  # Any arbitrary number
learning_rate = 0.1


dataX = []
dataY = []

for i in range(0, len(drum_data_diff_1d) - sequence_length):
    x_str = drum_data_diff_1d[i:i + sequence_length]
    y_str = drum_data_diff_1d[i + 1: i + sequence_length + 1]
    #print(i, x_str, '->', y_str)

    x = [drum_note_dic[c] for c in x_str]  # x str to index
    y = [drum_note_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

#X = tf.placeholder(tf.int32, [None, sequence_length])
#Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(dataX, num_classes)
print(X_one_hot)  # check out the shape



#########
# 옵션 설정
######
total_epoch = 20
batch_size = 100
learning_rate = 0.0002
# 신경망 레이어 구성 옵션
n_hidden = 256
n_input = 100
n_noise = 100 # 생성기의 입력값으로 사용할 노이즈의 크기



#########
# 신경망 모델 구성
######
# GAN 도 Unsupervised 학습이므로 Autoencoder 처럼 Y 를 사용하지 않습니다.
X = tf.placeholder(tf.float32, [None, n_input])
# 노이즈 Z를 입력값으로 사용합니다.
Z = tf.placeholder(tf.float32, [None, n_noise])

# 생성기 신경망에 사용하는 변수들입니다.
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 판별기 신경망에 사용하는 변수들입니다.
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
# 판별기의 최종 결과값은 얼마나 진짜와 가깝냐를 판단하는 한 개의 스칼라값입니다.
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


# 생성기(G) 신경망을 구성합니다.
def generator(noise_z):
    hidden_layer = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    generated_outputs = tf.sigmoid(tf.matmul(hidden_layer, G_W2) + G_b2)
    return generated_outputs


# 판별기(D) 신경망을 구성합니다.
def discriminator(inputs):
    hidden_layer = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    discrimination = tf.sigmoid(tf.matmul(hidden_layer, D_W2) + D_b2)
    return discrimination


# 랜덤한 노이즈(Z)를 만듭니다.
def get_noise(batch_size):
    return np.random.normal(size=(batch_size, n_noise))


# 노이즈를 이용해 랜덤한 이미지를 생성합니다.
G = generator(Z)
# 노이즈를 이용해 생성한 이미지가 진짜 이미지인지 판별한 값을 구합니다.
D_gene = discriminator(G)
# 진짜 이미지를 이용해 판별한 값을 구합니다.
D_real = discriminator(X)

# 논문에 따르면, GAN 모델의 최적화는 loss_G 와 loss_D 를 최대화 하는 것 입니다.
# 논문의 수식에 따른 다음 로직을 보면 loss_D 를 최대화하기 위해서는 D_gene 값을 최소화하게 됩니다.
# 판별기에 진짜 이미지를 넣었을 때에도 최대값을 : tf.log(D_real)
# 가짜 이미지를 넣었을 때에도 최대값을 : tf.log(1 - D_gene)
# 갖도록 학습시키기 때문입니다.
# 이것은 판별기는 생성기가 만들어낸 이미지가 가짜라고 판단하도록 판별기 신경망을 학습시킵니다.
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
# 반면 loss_G 를 최대화하기 위해서는 D_gene 값을 최대화하게 되는데,
# 이것은 가짜 이미지를 넣었을 때, 판별기가 최대한 실제 이미지라고 판단하도록 생성기 신경망을 학습시킵니다.
# 논문에서는 loss_D 와 같은 수식으로 최소화 하는 생성기를 찾지만,
# 결국 D_gene 값을 최대화하는 것이므로 다음과 같이 사용할 수 있습니다.
loss_G = tf.reduce_mean(tf.log(D_gene))

# loss_D 를 구할 때는 생성기 신경망에 사용되는 변수만 사용하고,
# loss_G 를 구할 때는 판별기 신경망에 사용되는 변수만 사용하여 최적화를 합니다.
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# GAN 논문의 수식에 따르면 loss 를 극대화 해야하지만, minimize 하는 최적화 함수를 사용하기 때문에
# 최적화 하려는 loss_D 와 loss_G 에 음수 부호를 붙여줍니다.
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = batch_size
loss_val_D, loss_val_G = 0, 0


for epoch in range(total_epoch):
    for i in range(10):
        x = dataX[i]
        batch_xs = np.asarray(x)
        noise = get_noise(1)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})


''' 
    sample_size = 10
    noise = get_noise(sample_size)

    samples = sess.run(G, feed_dict={Z: noise})

    fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

    for i in range(sample_size):
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(samples[i], (28, 28)))

    plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
'''

print('최적화 완료!')
