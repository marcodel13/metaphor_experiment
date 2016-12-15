from keras.layers import Embedding, Input, LSTM, Dense, GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model
import keras.preprocessing.sequence
import numpy as np
import pickle

input_length = 50
max_index = 16595
embeddings = pickle.load(open('glove.pickle','rb'))
hidden_size = 20
hidden = LSTM
batch_size = 200
epochs = 50
inputs = '_vuamc_shuffled.txt'
targets = '_targets_shuffled.txt'
train_embeddings = True         # set to False to keep glove embeddings
class_weights = {0:0.03, 1:0.97}
optimizer = 'adam'
# seed = 0


################################################
# MODEL

print("Generate model")

# embedding_weights = np.random.normal(size=max_index*50).reshape(max_index, 50)
embedding_weights = embeddings

# define layers
input_layer = Input(shape=(input_length,), name='input')
embeddings = Embedding(input_dim=max_index, output_dim=50, input_length=input_length, weights=None, name='embeddings', trainable=train_embeddings, mask_zero=True)(input_layer)
bidirectional = Bidirectional(hidden(hidden_size, return_sequences=True), name='RNN')(embeddings)
output_layer = TimeDistributed(Dense(1, activation='sigmoid'), name='output')(bidirectional)
# output_layer = Dense(1, activation='sigmoid', name='output')(bi_LSTM)

# define model and compile
model = Model(input=input_layer, output=output_layer)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy','precision', 'recall'], sample_weight_mode='temporal')


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
X_test, Y_test = X_padded[split:], Y_padded[split:]


################################################
# TRAIN AND TEST

print("Train model")
neg, pos = class_weights[1]-class_weights[0], class_weights[0]
sample_weight = Y_train.reshape(Y_train.shape[0:2])*neg + pos
model.fit(X_train, Y_train, validation_split=float(3.0/17), batch_size=batch_size, nb_epoch=epochs, verbose=1, sample_weight=sample_weight)

# TODO test if accuracy is correct for padding
# print test results
print('\n\n\n')
metrics = model.metrics_names
evaluation = model.evaluate(X_test, Y_test)

print('\t'.join(['%s: %f' % metric_value for metric_value in zip(metrics, evaluation)]))

# To print output on test:
# predictions = model.predict(X_test)
# print(predictions)

