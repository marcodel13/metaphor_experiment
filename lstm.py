from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from random import shuffle
import sklearn.metrics as sk

# ********* random initialization *********

# embeddings = tf.Variable(tf.random_normal([16594, 50]), name="embeddings")

# sess.run(embeddings)
# new_line = []
# for line in dataset:
#     # new_line = []
#     for element in line.split(','):
#         new_line.append(int(element))
#     sent_wembs = tf.nn.embedding_lookup(embeddings, new_line)
# print(sess.run(sent_wembs))


# ********* data *********

# glove vectors
glove_subsamp = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_glove.6B.50d_sampled_vuacm_voc_no_words_new_version_with_random_vector.txt').readlines()

# dataset
# dataset = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_vuamc_translated_into_a_list_of_lists_of_indexes_new_version_without_100000.txt').readlines()
# dataset = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/1_to_5/data_lenght_1_5.txt').readlines()
dataset = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_vuamc_translated_prova.txt').readlines()
# dataset = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/10_to_20/data_lenght_10_20.txt').readlines()
# dataset = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_vuamc_translated_prova.txt').readlines()
# labels
# labels = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_targets.txt').readlines()
# labels = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/1_to_5/target_lenght_1_5.txt').readlines()
labels = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_targets_prova.txt').readlines()

# labels = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/10_to_20/target_lenght_10_20.txt').readlines()
# labels = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_targets_prova.txt').readlines()

# file wtith the lenght of the sentences
# sequence_lenght = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_lenght_of_sequences.txt').readlines()
# sequence_lenght = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/1_to_5/seqlen_lenght_1_5.txt').readlines()
sequence_lenght = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_lenght_of_sequences_prova.txt').readlines()
# sequence_lenght = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/10_to_20/seqlen_lenght_10_20.txt').readlines()
# sequence_lenght = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/_lenght_of_sequences_prova.txt').readlines()

seq_max_len_or_n_steps = 5

# convert each string in glove in float
glove_with_int = []
for line in glove_subsamp:
    l = []
    for item in line.split():
        l.append(float(item))

    glove_with_int.append(l)

glove_matrix = tf.Variable(glove_with_int, name="glove")

data = [] # in this list are stored all the sentence
seq_len = []


for line in dataset:

    # line is a sequence of strings: create a list of lists of int
    new_line = []
    for element in line.split(','):
        new_line.append(int(element))

    # store the lenght of each sentence (before padding!)
    seq_len.append(len(new_line))

    # padding
    lenght_newline = len(new_line)
    while lenght_newline < seq_max_len_or_n_steps:
        new_line.append(16593) # the 16594 vector in the glove file is a vector of zeros
        lenght_newline = lenght_newline + 1

    # look up the glove embeddings using the index
    sentences_embeddings = []
    for index in new_line:
        # 100000 is the index assigned to words in vuamc that are not in glove: for this words create a random variable
        if index != 100000:
            # sent_wembs = tf.nn.embedding_lookup(glove_matrix, index)
            word_emb = tf.nn.embedding_lookup(glove_matrix, index)
            sentences_embeddings.append(word_emb)
        else:
            word_emb = tf.Variable(tf.random_normal([1, 50]))
            sentences_embeddings.append(word_emb)

    data.append(sentences_embeddings)



# convert each line of the labels into a list of lists of indexes
labels_int = []
for line in labels:
    l = []
    for item in line.split(', '):

        # version for one class
        item_list = []
        item_list.append(int(item))
        l.append(item_list)

        # version for two classes
        # if item == '0':
        #     item_list = []
        #     item_list.append(float(item))
        #     item_list.append(1.0)
        #     l.append(item_list)
        # if item == '1':
        #     item_list = []
        #     item_list.append(float(item))
        #     item_list.append(0.0)
        #     l.append(item_list)
    # padding also fot the labels
    lenght_l = len(l)
    while lenght_l < seq_max_len_or_n_steps:
        l.append([0])  # the 16594 vector in the glove file is a vector of zeros
        lenght_l = lenght_l + 1

    # l = [list(x) for x in l]
    labels_int.append(l)



# convert each string in seqlen in int
seqlen_int = []
for line in sequence_lenght:
    l = []
    for item in line.split():
        l.append(int(item))

    seqlen_int.append(l)

# need to flat the nested lists in seqlen
seqlen_int = sum(seqlen_int, [])


# ********* model *********


# train and test


# Parameters
learning_rate = 0.001
batch_size = 32
display_step = 10
vector_dimension = 50
num_epochs = 30
threshold = 0.5


# Network Parameters
# seq_max_len_or_n_steps --> defined above = 117
n_hidden = 64
n_classes = 1
test_size = 32


# placeholders
x = tf.placeholder("float", [None, seq_max_len_or_n_steps, vector_dimension])

# 27/11
# the original version of the y is: y = tf.placeholder("float", [batch_size, n_classes]). in this case you have a matrix in which the number of rows is given by the number of elements in the batch
# this makes perfect sense, because in the original setting we were considering only one output per sentence, and therefore the number of outputs were equal to the number of elements in the batc
# in the new formulation of the y we have to multiply the size if the batch (i.e. the number of sentences) by the number of words or each sentence (seq_max_len_or_n_steps)

# y = tf.placeholder("float", [batch_size*seq_max_len_or_n_steps, n_classes])
# 28/11
# what I wrote above is not completely true: I think for y it is better to have a batch_sizeXseq_max_len_or_n_stepsXn_classes matrix instead of a batch_size*seq_max_len_or_n_stepsXn_classes:
# the difference is that the latter is a 2-d matrix in which each rows corresponds to a word, but there is no distinction between sentences
# the former, is a 3-d matrix in which you distinguish between different rows
y = tf.placeholder("float", [None, seq_max_len_or_n_steps, n_classes])

mask = tf.placeholder("float", [None, seq_max_len_or_n_steps])

seqlen = tf.placeholder(tf.int32, [None])

# Define weights and biases
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}



# --------

def DynBiRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, vector_dimension])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, seq_max_len_or_n_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    try:
        outputs, states_fw, states_bw = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32, sequence_length=seqlen)

        # ************** IMP **************
        # if you roint the output you get:
        # [<tf.Tensor 'concat:0' shape=(3, 128) dtype=float32>, <tf.Tensor 'concat_1:0' shape=(3, 128) dtype=float32>, <tf.Tensor 'concat_2:0' shape=(3, 128) dtype=float32>]
        # this is what (I think) is happening: you have three sentences in input, each of them with three words. Each of the <tf.Tensor 'concat:0' shape=(3, 128) dtype=float32>
        # in the line above is a 2d matrix representing a sentence, and the dimension is (3, 128): 3 because three are the words for each sentence, and 128 is the dimensionality of each vector.
        # thi dimensionality is givent buy the fact that the hidden layer is 64, and since you have forward and backward 62*2=128.
        # so concluding, these vectors are the words you will need to pass to the classifier to decide lit/met
        # ***********************************

        # print(outputs)
        # print(outputs[-1])
        # print(weights)
        # print(states_fw)
        # print(states_fw)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32, sequence_length=seqlen)
    # ************** PHONG **************
    # outputs = tf.pack(outputs)
    # the pack in the line above converts a tensor array into a tensor. then, you create a 2-d matrix as shown below.
    # e.g: you have a tensor[2,4,d] (where d is the dimansion of the vectors), you pack it and then reshape it and get a matrix [8,d] that you use for classification
    # think about it: of you have a tensor [2,4,d] it mean you have a batch with two sentences and each sentence has four words, and therefore 8 words overall.
    # when you do wha phong suggests, you end up with a 2-d matrix that is 8 rows, i.e. one row for word (and each one has dimensionality = vector)! this makes perfect sense
    # matrix = tf.reshape(outputs, [-1, d] )
    # ***********************************

    # Linear activation, using rnn inner loop last output

    # ************** IMP **************
    # in the original script, the prediction is based on outputs[-1], i.e. the last vector of the sequence.
    # in my case I am not going to look at such a vector, but to all the vectors in output. How to do that? following what Phong suggests, that is:
    # you have the output that has shape  (3,3,128). you pack it, and it still will be (3,3,128), and then you reshape it and get a 2-d matrix (9,128). What has happened?
    # simply that you have converted a 3-d matrix with 3 sentences in it and where each sentence has 3 words into a 2-d matrix iwith 9 words.
    # now, given that each row in the matrix (i.e. row) has the same number of dimensions of the the weights matrix (128), you can multiply the reshaped matrix by the weights
    # ***********************************
    outputs = tf.pack(outputs)
    # print(outputs)
    matrix_of_results = tf.reshape(outputs, [-1, 2*n_hidden])
    # print(matrix)

    # otiginal version, in which you take only the last vector and multiply it by the last output (output[-1])

    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.matmul(matrix_of_results, weights['out']) + biases['out']

pred = DynBiRNN(x, seqlen, weights, biases)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred,tf.reshape(y, [-1, n_classes])))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# correct_pred = tf.equal(tf.to_float(thresholded_pred), tf.reshape(y, [-1, n_classes]))
thresholded_pred = tf.greater(pred, threshold)
thresholded_pred = tf.to_float(thresholded_pred)

correct_pred = tf.equal(thresholded_pred, tf.reshape(y, [-1, n_classes]))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # note: tf.cast just casts a tensor to a new type (in this cas e)tf.float32
                                                             # note: tf.reduce_mean computes the mean value

#*******
def compute_prec_rec_f1(my_prediction, gold_target):

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    m = 0
    while m < len(gold_target):
        if int(my_prediction[m]) == 1:
            if int(my_prediction[m]) == gold_target[m]:
                true_positives.append('o')
            if int(my_prediction[m]) != gold_target[m]:
                false_positives.append('o')
        else:
            if int(my_prediction[m]) == gold_target[m]:
                true_negatives.append('o')
            if int(my_prediction[m]) != gold_target[m]:
                false_negatives.append('o')
        m = m + 1

    print('TP:', len(true_positives), 'TN:', len(true_negatives), 'FP:', len(false_positives), 'FN:',
          len(false_negatives))
    acc_recalculated = round(float(len(true_positives) + len(true_negatives)) / (
        len(true_positives) + len(true_negatives) + len(false_positives) + len(false_negatives)), 3)

    try:
        precision = round(float(len(true_positives) / (len(true_positives) + len(false_positives))), 3)
        # print(precision)
    except (ZeroDivisionError, ValueError, NameError):
        print('no prec')

    try:
        recall = round(float(len(true_positives) / (len(true_positives) + len(false_negatives))), 3)
        # print(recall)
    except (ZeroDivisionError, ValueError, NameError):
        print('no rec')

    try:
        f_score = round(2 * (precision * recall) / (precision + recall), 3)
        # print(f_score)
    except (ZeroDivisionError, ValueError, NameError):
        print('no f-score')


    try:
        print('acc', acc_recalculated)
    except (ZeroDivisionError, ValueError, NameError):
        # continue
        print('')

    try:
        print('pre', precision)
    except (ZeroDivisionError, ValueError, NameError):
        # continue
        print('')

    try:
        print('rec', recall)
    except (ZeroDivisionError, ValueError, NameError):
        # continue
        print('')
    try:
        print('fscore', f_score)
    except (ZeroDivisionError, ValueError, NameError):
        # continue
        print('')





# *******



# Initializing the variables
init = tf.initialize_all_variables()



with tf.Session() as sess:
    sess.run(init)

    data = sess.run(data)
    lenght_data = len(data)

    # perc_train = 80


    labels_int = sess.run(tf.convert_to_tensor(labels_int))
    seqlen_int = sess.run(tf.convert_to_tensor(seqlen_int))

    # train
    train_data = data[:4200]
    print(len(train_data))
    train_data = np.array(train_data)
    train_labels = labels_int[:4200]
    train_seqlen = seqlen_int[:4200]

    train_indexes = np.arange(len(train_data))
    shuffle(train_indexes)

    # validation
    validation_data = data[4200:4300]
    validation_labels = labels_int[4200:4300]
    validation_seqlen = seqlen_int[4200:4300]

    validation_indexes = np.arange(len(validation_data))
    shuffle(validation_indexes)

    # test
    test_data = data[4300:4589]
    test_labels = labels_int[4300:4589]
    test_seqlen = seqlen_int[4300:4589]


    # train

    no_of_batches = int(len(train_data)/ batch_size)
    for i in range(num_epochs):
        # print(i)
        k = 0

        for j in range(no_of_batches):

            # print(j)
            batch_x = train_data[train_indexes[k:k+5]]
            batch_y = train_labels[train_indexes[k:k+5]]
            batch_seqlen = train_seqlen[train_indexes[k:k+5]]

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})

            # print(str(j) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

            k += 5

        # print("Epoch - ", str(i))

        acc = sess.run(accuracy, feed_dict={x: validation_data, y: validation_labels, seqlen: validation_seqlen})
        print("Epoch - ", str(i) + ", Validation Accuracy= " + "{:.5f}".format(acc))

        prediction_val = thresholded_pred.eval(feed_dict={x: validation_data, y: validation_labels, seqlen: validation_seqlen})
        target_val = [item for sublist in validation_labels for item in sublist]

        # compute accuracy, precision, recall and fscore for met class
        compute_prec_rec_f1(prediction_val, target_val)

        # true_positives = []
        # true_negatives = []
        # false_positives = []
        # false_negatives = []
        #
        # m = 0
        # while m < len(target):
        #     if int(prediction[m]) == 1:
        #         if int(prediction[m]) == target[m]:
        #             true_positives.append('o')
        #         if int(prediction[m]) != target[m]:
        #             false_positives.append('o')
        #     else:
        #         if int(prediction[m]) == target[m]:
        #             true_negatives.append('o')
        #         if int(prediction[m]) != target[m]:
        #             false_negatives.append('o')
        #     m = m + 1
        #
        # print('TP:', len(true_positives), 'TN:', len(true_negatives), 'FP:', len(false_positives), 'FN:', len(false_negatives))
        # acc_recalculated = round(float(len(true_positives) + len(true_negatives)) / (
        # len(true_positives) + len(true_negatives) + len(false_positives) + len(false_negatives)), 3)
        #
        # try:
        #     precision = round(float(len(true_positives) / (len(true_positives) + len(false_positives))), 3)
        #     # print(precision)
        # except (ZeroDivisionError, ValueError, NameError):
        #     print('no prec')
        #
        # try:
        #     recall = round(float(len(true_positives) / (len(true_positives) + len(false_negatives))), 3)
        #     # print(recall)
        # except (ZeroDivisionError, ValueError, NameError):
        #     print('no rec')
        #
        # try:
        #     f_score = round(2 * (precision * recall) / (precision + recall), 3)
        #     # print(f_score)
        # except (ZeroDivisionError, ValueError, NameError):
        #     print('no f-score')
        #
        #
        #
        # try:
        #     print('acc',acc_recalculated)
        # except (ZeroDivisionError, ValueError, NameError):
        #     continue
        #
        # try:
        #     print('pre',precision)
        # except (ZeroDivisionError, ValueError, NameError):
        #     continue
        #
        # try:
        #     print('rec',recall)
        # except (ZeroDivisionError, ValueError, NameError):
        #     continue
        #
        # try:
        #     print('fscore',f_score)
        # except (ZeroDivisionError, ValueError, NameError):
        #     continue
    # test

    print("Test Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_labels, seqlen: test_seqlen}))

    prediction_test = thresholded_pred.eval(feed_dict={x: test_data, y: test_labels, seqlen: test_seqlen})
    target_test = [item for sublist in test_labels for item in sublist]

    # compute accuracy, precision, recall and fscore for met class
    compute_prec_rec_f1(prediction_test, target_test)




# ********* graph execution *********


# with tf.Session() as sess:
#     sess.run(init)
#
#     step = 1
#     data = sess.run(data)
#     # print(data)
#
#     labels_int = sess.run(tf.convert_to_tensor(labels_int))
#     seqlen_int = sess.run(tf.convert_to_tensor(seqlen_int))
#
#
#     while step < training_iters:
#
#         print(step)
#     #
#     #     # batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
#     #     # print(batch_x)
#     #     # print(batch_y)
#     #     # print(batch_seqlen)
#     #     # Run optimization op (backprop)
#     #
#         # sess.run(optimizer, feed_dict={x: data, y: [[1], [0], [1], [1], [0], [1], [1], [0], [1]], seqlen: [3, 3, 2]})
#         sess.run(optimizer, feed_dict={x: data, y: labels_int, seqlen: seqlen_int})
# #
# #
# #
# #         # if step % display_step == 0: #questo vuol dire solo: se 'step' è un multiplo di display_step : non è per niente importante, potrebbe benissimo non esserci
# #         if step % 1 == 0:  # questo vuol dire solo: se 'step' è un multiplo di display_step : non è per niente importante, potrebbe benissimo non esserci
# #
# #             # Calculate batch accuracy
# #             acc = sess.run(accuracy, feed_dict={x: data, y: [[1], [1], [1]], seqlen: [2, 2, 1]})
# #             # Calculate batch loss
# #             loss = sess.run(cost, feed_dict={x: data, y: [[1], [1], [1]], seqlen: [2, 2, 1]})
# #             # print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
# #
#         step += 1
# #
# #         # print("Optimization Finished!")
# #
#         Calculate accuracy
#         test_data = testset.data
# #         # test_label = testset.labels
# #         # test_seqlen = testset.seqlen
# #         # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: data, y: [[1], [1], [1]], seqlen: [2, 2, 1]}))