from keras.layers import Embedding, Input, LSTM, Dense
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Model
import keras.preprocessing.sequence
import numpy as np

input_length = 50
max_index = 20000
# embeddings = filename        #TODO fetch glove embeddings from somewhere
LSTM_size = 20
batch_size = 20
epochs=5
class_weight = {0:0.1, 1:0.9}
training_inputs = '_vuamc_shuffled.txt'
training_targets = '_targets_shuffled.txt'
train_embeddings = True         # set to False to keep glove embeddings


################################################
# MODEL

print("Generate model")

# TODO change for glove embeddings
# create embeddings matrix
embedding_weights = np.random.normal(size=max_index*50).reshape(max_index, 50)

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

# read in training data from file
print("Read training data")
X_train = [[int(index) for index in line.split(',')] for line in open(training_inputs).readlines()]
Y_train = [[[int(index)] for index in line.split(',')] for line in open(training_targets).readlines()]

# Pad training data
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train, dtype='int32', maxlen=input_length)
Y_train_padded = keras.preprocessing.sequence.pad_sequences(Y_train, dtype='int32', maxlen=input_length)

# Pad test data
# X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test, dtype='int32', maxlen=input_length)
# Y_test_padded = keras.preprocessing.sequence.pad_sequences(Y_test, dtype='int32', maxlen=input_length)



################################################
# TRAIN AND TEST

# TODO use sample weight to weight models
print("Train model")
model.fit(X_train_padded, Y_train_padded, batch_size=batch_size, nb_epoch=epochs, verbose=1)

# TODO get test data
# TODO implement precision and recall in keras
# TODO test if accuracy is correct for padding
# model.evaluate(X_test, Y_test)
