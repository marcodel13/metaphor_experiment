from keras.layers import Embedding, Input, LSTM, Dense
from keras.models import Model
from keras.wrappers import Bidirectional
from keras

input_length = 50
max_index = 20000
glove = glove_embeddings        #TODO fetch from somewhere
LSTM_size = 20
batch_size=20
epochs=10


# define layers
input_layer = Input(shape=(input_length,), name='input')
embeddings = Embedding(input_dim=max_index, output_dim=50, input_length=input_length, weights=[glove], dtype='int32', name='embeddings', trainable=False, mask_zero=True)(input_layer)
bi_LSTM = Bidirectional(LSTM(LSTM_size), name='RNN')(embeddings)
output_layer = Dense(1, activation='sigmoid')

# define model and compile
model = Model(input=input_layer, output=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   #TODO would liket o add also precision and recall


# define train data and test data


# pad training datao
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train, dtype='int32', maxlen=input_length)
Y_train_padded = keras.preprocessing.sequence.pad_sequences(Y_train, dtype='int32', maxlen=input_length)

model.fit(X_train_padded, Y_train_padded, batch_size=batch_size, nb_epochs=epochs)



