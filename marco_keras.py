from keras.layers import Embedding, Input, LSTM, Dense
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model
import keras.preprocessing.sequence
import numpy as np
import pickle

input_length = 50
max_index = 16595
embeddings = pickle.load(open('glove.pickle','rb'))
LSTM_size = 20
batch_size = 20
epochs=5
class_weight = {0:0.1, 1:0.9}
inputs = '_vuamc_shuffled.txt'
targets = '_targets_shuffled.txt'
train_embeddings = False         # set to False to keep glove embeddings


################################################
# MODEL

print("Generate model")

# embedding_weights = np.random.normal(size=max_index*50).reshape(max_index, 50)
embedding_weights = embeddings

# define layers
input_layer = Input(shape=(input_length,), name='input')
embeddings = Embedding(input_dim=max_index, output_dim=50, input_length=input_length, weights=None, name='embeddings', trainable=train_embeddings, mask_zero=True)(input_layer)
bi_LSTM = Bidirectional(LSTM(LSTM_size, return_sequences=True), name='RNN')(embeddings)
output_layer = TimeDistributed(Dense(1, activation='sigmoid'), name='output')(bi_LSTM)
# output_layer = Dense(1, activation='sigmoid', name='output')(bi_LSTM)

# define model and compile
model = Model(input=input_layer, output=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   #TODO would liket o add also precision and recall


################################################
# DATA

# read in data
X = [[int(index) for index in line.split(',')] for line in open(inputs).readlines()]
Y = [[[int(index)] for index in line.split(',')] for line in open(targets).readlines()]

# pad data
X_padded = keras.preprocessing.sequence.pad_sequences(X, dtype='int32', maxlen=input_length)
Y_padded = keras.preprocessing.sequence.pad_sequences(Y, dtype='int32', maxlen=input_length)

assert X_padded.shape == Y_padded.shape[0:2], "input and output shapes do not match, shape in = %s, shape out = %s" % (X_padded.shape, Y_padded.shape)

split = int(0.85*X_padded.shape[0])

X_train, Y_train = X_padded[:split], Y_padded[:split]
X_test, Y_test = X_padded[split:], Y_padded[split]


################################################
# TRAIN AND TEST

# TODO use sample weight to weight models
print("Train model")
model.fit(X_train, Y_train, validation_split=float(3.0/17), batch_size=batch_size, nb_epoch=epochs, verbose=1)

# TODO implement precision and recall in keras
# TODO test if accuracy is correct for padding
model.evaluate(X_test, Y_test)
