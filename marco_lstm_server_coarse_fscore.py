from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from random import shuffle
import time

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


# ****************** DATA FILES ******************

glove_subsamp = open('/home/mdtredi1/my_virtual_env/data/_glove.6B.50d_sampled_vuacm_voc_no_words_new_version_with_random_vector.txt').readlines()

# dataset
dataset = open('/home/mdtredi1/my_virtual_env/data/_vuamc_shuffled.txt').readlines()
# dataset = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/1_to_5/data_lenght_1_5.txt').readlines()
# dataset = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/10_to_20/data_lenght_10_20.txt').readlines()

# labels
labels = open('/home/mdtredi1/my_virtual_env/data/_targets_shuffled.txt').readlines()
# labels = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/1_to_5/target_lenght_1_5.txt').readlines()
# labels = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/10_to_20/target_lenght_10_20.txt').readlines()

# file wtith the lenght of the sentences
sequence_lenght = open('/home/mdtredi1/my_virtual_env/data/_lenght_of_sequences_shuffled.txt').readlines()
# sequence_lenght = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/1_to_5/seqlen_lenght_1_5.txt').readlines()
# sequence_lenght = open('/Users/marcodeltredici/workspace/PYCHARM/metaphor_and_generalization_second_act/LSTM/run_lstm/data/batches_divided_on_lenght/10_to_20/seqlen_lenght_10_20.txt').readlines()

results = open('/home/mdtredi1/my_virtual_env/data/log_coarse_fscore.txt','a')

seq_max_len_or_n_steps = 117

# ****************** MANIPULATE THE DATA ******************

# convert each string in glove in float
glove_with_int = []
for line in glove_subsamp:
    l = []
    for item in line.split():
        l.append(float(item))

    glove_with_int.append(l)

glove_matrix = tf.Variable(glove_with_int, name="glove") # ok


# padding and convert line as sequence of strings into line as list of integers
def manipulate_data(dataset):

    data = []
    # for line in dataset:
    #
        # # line is a sequence of strings: create a list of lists of int
        # new_line = []
        # for element in line.split(','):
        #     new_line.append(int(element))
        #
        # # padding
        # lenght_newline = len(new_line)
        # while lenght_newline < seq_max_len_or_n_steps:
        #     new_line.append(16593)  # the 16594 vector in the glove file is a vector of zeros
        #     lenght_newline = lenght_newline + 1


    for line in dataset:

        paddings = [[0, seq_max_len_or_n_steps-len(line.split(','))]]
        new_line = list(map(int, line.split(',')))
        new_line = tf.pad(new_line, paddings)
        sentences_embeddings = tf.nn.embedding_lookup(glove_matrix, new_line)


        data.append(sentences_embeddings)
    return data
data = manipulate_data(dataset) # ok


def manipulate_labels(labels):

    # # convert each line of the labels into a list of lists of indexes
    # labels_int = []
    # for line in labels:
    #     l = []
    #     for item in line.split(', '):
    #
    #         # version for one class
    #         item_list = []
    #         item_list.append(int(item))
    #         l.append(item_list)
    #
    #
    #     # padding also fot the labels
    #     lenght_l = len(l)
    #     while lenght_l < seq_max_len_or_n_steps:
    #         l.append([0])  # the 16594 vector in the glove file is a vector of zeros
    #         lenght_l = lenght_l + 1
    #     #
    #     # l = [list(x) for x in l]
    #     labels_int.append(l)

    labels_int = []
    for line in labels:
        paddings = [[0, seq_max_len_or_n_steps - len(line.split(','))]]
        new_line = list(map(int, line.split(',')))
        new_line = tf.pad(new_line, paddings)
        new_line = tf.reshape(new_line, [seq_max_len_or_n_steps, 1])
        # print('new_line', new_line)
        labels_int.append(new_line)


    return labels_int
labels_int = manipulate_labels(labels) # ok


def manipulate_seqlen(sequence_lenght):

    # convert each string in seqlen in int
    seqlen_int = []
    for line in sequence_lenght:
        l = []
        for item in line.split():
            l.append(int(item))

        seqlen_int.append(l)
    seqlen_int = sum(seqlen_int, [])
    return seqlen_int

seqlen_int = manipulate_seqlen(sequence_lenght) # ok

# ****************** SPLIT THE DATA ******************

def size_train_val_test(p_train, p_val, p_test):
    # train_data = data[:80]
    # train_labels = labels_int[:80]  # till 11,000
    # train_seqlen = seqlen_int[:80]
    #
    # validation_data = data[81:100]
    # validation_labels = labels_int[81:100]  # till 11,000
    # validation_seqlen = seqlen_int[81:100]
    #
    # test_data = data[100:132]
    # test_labels = labels_int[100:132]  # everything beyond 10,000
    # test_seqlen = seqlen_int[100:132]
    lenght_data = len(data)
    # print(lenght_data)
    perc_train = int((lenght_data * p_train) / 100)
    # print(perc_train)
    perc_val = int((lenght_data * p_val) / 100)
    # print(int(perc_val))
    perc_test = int((lenght_data * p_test) / 100)

    train_data = data[:perc_train]
    train_labels = labels_int[:perc_train]
    train_seqlen = seqlen_int[:perc_train]

    validation_data = data[perc_train:perc_train+int(perc_val)]
    validation_labels = labels_int[perc_train:perc_train+int(perc_val)]
    validation_seqlen = seqlen_int[perc_train:perc_train+int(perc_val)]

    test_data = data[-perc_test:]
    test_labels = labels_int[-perc_test:]
    test_seqlen = seqlen_int[-perc_test:]

    return train_data, train_labels, train_seqlen, validation_data, validation_labels, validation_seqlen, test_data, test_labels, test_seqlen

train_data, train_labels, train_seqlen, validation_data, validation_labels, validation_seqlen, test_data, test_labels, test_seqlen = size_train_val_test(70, 15, 15) # ok



# ****************** PARAMETERS ******************

# Parameters
learning_rate = 0.01
batch_size = 32
vector_dimension = 50
no_of_batches = int(len(train_data) / batch_size)
num_epochs = 10



# Network Parameters
# seq_max_len_or_n_steps --> defined above
n_hidden = 64 # check it
n_classes = 1 # lit/met

# placeholders
x = tf.placeholder("float", [None, seq_max_len_or_n_steps, vector_dimension])
y = tf.placeholder("float", [None, seq_max_len_or_n_steps, n_classes])
mask = tf.placeholder("float", [None, seq_max_len_or_n_steps])

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

threshold = tf.placeholder(tf.float32, None)
# threshold = 0.5 # check
weight_for_cost = tf.placeholder(tf.float32, None)


# Define weights and biases
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# ****************** DEFINE THE MODEL ******************

def DynBiRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # print('x initial', x)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # print('x transpose', x)
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, vector_dimension])
    # print('x reshape', x)
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, seq_max_len_or_n_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    try:
        outputs, states_fw, states_bw = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32, sequence_length=seqlen)

        # ************** IMP **************
        # if you print the output you get:
        # [<tf.Tensor 'concat:0' shape=(3, 128) dtype=float32>, <tf.Tensor 'concat_1:0' shape=(3, 128) dtype=float32>, <tf.Tensor 'concat_2:0' shape=(3, 128) dtype=float32>]
        # this is what (I think) is happening: you have three sentences in input, each of them with three words. Each of the <tf.Tensor 'concat:0' shape=(3, 128) dtype=float32>
        # in the line above is a 2d matrix representing a sentence, and the dimension is (3, 128): 3 because three are the words for each sentence, and 128 is the dimensionality of each vector.
        # thi dimensionality is givent by the fact that the hidden layer is 64, and since you have forward and backward 62*2=128.
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
    # think about it: if you have a tensor [2,4,d] it mean you have a batch with two sentences and each sentence has four words, and therefore 8 words overall.
    # when you do what phong suggests, you end up with a 2-d matrix that is 8 rows, i.e. one row for word (and each one has dimensionality = vector)! this makes perfect sense
    # matrix = tf.reshape(outputs, [-1, d] )
    # ***********************************

    # Linear activation, using rnn inner loop last output

    # ************** IMP **************
    # in the original script, the prediction is based on outputs[-1], i.e. the last vector of the sequence.
    # in my case I am not going to look at such vector, but to all the vectors in output. How to do that? following what Phong suggests, that is:
    # you have the output that has shape  (3,3,128). you pack it, and it still will be (3,3,128), and then you reshape it and get a 2-d matrix (9,128). What has happened?
    # simply that you have converted a 3-d matrix with 3 sentences in it and where each sentence has 3 words into a 2-d matrix with 9 words.
    # now, given that each row in the matrix (i.e. row) has the same number of dimensions of the the weights matrix (128), you can multiply the reshaped matrix by the weights
    # ***********************************
    # print('outputs', outputs)
    outputs = tf.pack(outputs)
    # print('outputs_pack', outputs)
    matrix_of_results = tf.reshape(outputs, [-1, 2*n_hidden])
    # print('matrix_of_results', matrix_of_results)


    # otiginal version, in which you take only the last vector and multiply it by the last output (output[-1])

    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return tf.matmul(matrix_of_results, weights['out']) + biases['out']


pred = DynBiRNN(x, seqlen, weights, biases)


# ****************** EVALUATE THE MODEL ******************

# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred,tf.reshape(y, [-1, n_classes])))
# new cost function

cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(pred,tf.reshape(y, [-1, n_classes]), weight_for_cost))

# check if cross entrpy is fine
# cost is still computed on sigmoided value, not on thresholded one

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


thresholded_pred = tf.greater(pred, threshold)
thresholded_pred = tf.to_float(thresholded_pred)


check_pred = tf.equal(thresholded_pred, tf.reshape(y, [-1, n_classes]))

# train
check_pred_reshape_train = tf.reshape(check_pred, [batch_size, seq_max_len_or_n_steps])
masked_pred_train =  tf.mul(tf.to_float(check_pred_reshape_train),mask)
accuracy = tf.reduce_sum(masked_pred_train) / tf.reduce_sum(tf.to_float(seqlen))


# validation
check_pred_reshape_val = tf.reshape(check_pred, [len(validation_data), seq_max_len_or_n_steps])
masked_pred_val =  tf.mul(tf.to_float(check_pred_reshape_val),mask)
accuracy_val = tf.reduce_sum(masked_pred_val) / tf.reduce_sum(tf.to_float(seqlen))


# test
check_pred_reshape_test = tf.reshape(check_pred, [len(test_data), seq_max_len_or_n_steps])
masked_pred_test =  tf.mul(tf.to_float(check_pred_reshape_test),mask)
accuracy_test = tf.reduce_sum(masked_pred_test) / tf.reduce_sum(tf.to_float(seqlen))


# ****************** FUNCTIONS ******************

# function to create the mask
def define_mask(number_of_sequences, bsl, max_lenght):

    m = np.zeros((number_of_sequences, max_lenght), dtype=np.int32)

    i = 0
    while i < len(bsl):

        m[i][0:bsl[i]] = 1
        i = i+1

    return m

# function that computes acc, prec, rec, f1
def compute_prec_rec_f1_new(thresholded_prediction_reshaped_masked, labels_reshaped, data, seq_max_len_or_n_steps,
                            print_tp_etc, print_prec, print_rec, print_f1):


    true_positives = tf.reduce_sum(tf.mul(thresholded_prediction_reshaped_masked, tf.to_float(labels_reshaped)))

    true_positives = tf.cast(true_positives, tf.int32)

    # print('tp', true_positives.eval())

    # FP
    a = tf.reshape(thresholded_prediction_reshaped_masked, [1, len(data)* seq_max_len_or_n_steps])
    b = tf.reshape(labels_reshaped, [1, len(data) * seq_max_len_or_n_steps])

    subtraction = tf.sub(a, tf.to_float(b))

    # print(a.eval())
    # print(b.eval())
    subtraction = subtraction.eval().tolist()
    subtraction = sum(subtraction,[])

    false_positives = subtraction.count(1.0)
    # print('fp',false_positives)

    false_negatives = subtraction.count(-1.0)
    # print('fn',false_negative)

    if print_tp_etc:
        print('tp', true_positives.eval())
        print('fp', false_positives)
        print('fn', false_negatives)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2 * (precision * recall) / (precision + recall)

    # try:
    #     precision = true_positives / (true_positives + false_positives)
    #     if print_prec:
    #         print('pre', precision.eval())
    # except (ZeroDivisionError, ValueError, NameError):
    #     print('no prec')
    #
    # try:
    #     recall = true_positives / (true_positives + false_negatives)
    #     if print_rec:
    #         print('rec', recall.eval())
    # except (ZeroDivisionError, ValueError, NameError):
    #     print('no rec')
    #
    # try:
    #     f_score = 2 * (precision * recall) / (precision + recall)
    #     if print_f1:
    #         print('fscore', f_score.eval())
    # except (ZeroDivisionError, ValueError, NameError):
    #     print('no f-score')

    return tf.to_float(f_score), true_positives,  false_positives, false_negatives


# function to output number of met and lit in dataset
def print_lit_and_met_in_dataset(subset, labels, seqlen):
    met = []
    for line in labels:

        for element in line:
            if 1 in element:
                met.append(element)

    print('total number of word in ', subset, sum(seqlen))
    print('literal occurrences in ', subset, sum(seqlen) - len(met))
    print('metaphorical occurrences in ', subset, len(met))


# ****************** RUN THE MODEL ******************

# Initializing the variables
init = tf.initialize_all_variables()


with tf.Session() as sess:

    sess.run(init)

    # ************* DATA *************

    data = sess.run(data)

    labels_int = sess.run(tf.convert_to_tensor(labels_int))
    seqlen_int = sess.run(tf.convert_to_tensor(seqlen_int))

    # train
    train_data = sess.run(tf.convert_to_tensor(train_data))
    train_data = np.array(train_data)
    train_labels = sess.run(tf.convert_to_tensor(train_labels))
    train_seqlen = sess.run(tf.convert_to_tensor(train_seqlen))

    train_indexes = np.arange(len(train_data))
    shuffle(train_indexes)

    # validation
    validation_data = sess.run(tf.convert_to_tensor(validation_data))
    validation_labels = sess.run(tf.convert_to_tensor(validation_labels))
    # print(validation_labels)
    validation_seqlen = sess.run(tf.convert_to_tensor(validation_seqlen))
    # print(validation_seqlen)

    # validation_indexes = np.arange(len(validation_data))
    # shuffle(validation_indexes)

    # test
    test_data = sess.run(tf.convert_to_tensor(test_data))
    test_labels = sess.run(tf.convert_to_tensor(test_labels))
    test_seqlen = sess.run(tf.convert_to_tensor(test_seqlen))

    # print number of literal and metaphorical words in each subsample
    print_lit_and_met_in_dataset('validation', validation_labels, validation_seqlen)
    print_lit_and_met_in_dataset('test', test_labels, test_seqlen)

    # ************* TRAINING *************

    best_params = [0,0]
    best_fscore = 0
    best_tp = 0
    best_fp = 0
    best_fn = 0

    max_pen = 100
    step_pen = 5

    counter = 1

    for pen in np.arange(1, max_pen, step_pen):
        start_time = time.time()

        print('\npen', pen)


        for i in range(num_epochs): # from 0 to 29
            # print('epoch ', i)
            k = 0

            for j in range(no_of_batches):

                batch_x = train_data[train_indexes[k:k+batch_size]]
                batch_y = train_labels[train_indexes[k:k+batch_size]]
                batch_seqlen = train_seqlen[train_indexes[k:k+batch_size]]

                # define the mask
                m_train = define_mask(batch_size, batch_seqlen, seq_max_len_or_n_steps)

                _, final_accuracy, thresholded_prediction, check_the_prediction, masked_prediction = \
                    sess.run([optimizer, accuracy, thresholded_pred, check_pred_reshape_train, masked_pred_train],
                             feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, mask: m_train, threshold: 0.0, weight_for_cost: pen})

                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen, mask: m_train, weight_for_cost: pen})
                # print('loss',loss)

                k += 5


            # ************* VALIDATION *************


        for thresh in np.arange(0.0, 1.0, 0.05):

            print('\nthresh', thresh)
            # define the mask
            m_val = define_mask(len(validation_data), validation_seqlen, seq_max_len_or_n_steps)

            # acc = sess.run(accuracy_val, feed_dict={x: validation_data, y: validation_labels, seqlen: validation_seqlen, mask: m_val, threshold: thresh})

            # print("Validation Accuracy= " + "{:.5f}".format(acc))


            thresholded_prediction_val = sess.run(thresholded_pred,feed_dict={x: validation_data, y: validation_labels, seqlen: validation_seqlen, mask: m_val, threshold: thresh})
            thresholded_prediction_val_reshaped = tf.reshape(thresholded_prediction_val,[len(validation_data), seq_max_len_or_n_steps])
            thresholded_prediction_val_reshaped_masked = tf.mul(thresholded_prediction_val_reshaped, m_val)

            val_labels_reshaped = (tf.reshape(validation_labels, [len(validation_data), seq_max_len_or_n_steps]))

            fscore, tp, fp, fn = compute_prec_rec_f1_new(thresholded_prediction_val_reshaped_masked, val_labels_reshaped, validation_data,
                                    seq_max_len_or_n_steps,
                                    print_tp_etc=True, print_prec=True, print_rec=True, print_f1=True)



            fscore = float(fscore.eval())
            tp = int(tp.eval())




            if fscore > best_fscore:
                best_fscore = fscore
                best_params[0] = pen
                best_params[1] = thresh
                best_tp = tp
                best_fp = fp
                best_fn = fn
                print("\nnew best_fscore: " + str(best_fscore))
                print("new best_tp: " + str(best_tp))
                print("new best_fp: " + str(best_fp))
                print("new best_fn: " + str(best_fn))
                print("pen: " + str(best_params[0]))
                print("thresh: " + str(best_params[1]))

                results.write("new best_fscore " + str(thresh) + ' ' + str(best_fscore) + '\n')


        print("\nAFTER PEN=", pen)
        print("best_fscore: " + str(best_fscore))
        print("best_tp: " + str(best_tp))
        print("best_fp: " + str(best_fp))
        print("best_fn: " + str(best_fn))
        print("pen: " + str(best_params[0]))
        print("thresh: " + str(best_params[1]))
        elapsed_time = time.time() - start_time
        print('elapsed_time:', round(elapsed_time / 60, 2), 'runs left:', (max_pen / step_pen) - counter, '\n **************\n')
        results.write('\n' + "AFTER PEN=" + str(pen) + "best_fscore: " + str(best_fscore)+ ' ' + "pen: " + str(best_params[0]) + "thresh: " + str(best_params[1]) + '\n')

        counter = counter + 1


    print("\nFINAL REUSLTS")
    print('best_params', best_params)
    print('best_fscore', best_fscore)
    print("best_tp: " + str(best_tp))
    print("best_fp: " + str(best_fp))
    print("best_fn: " + str(best_fn))
    results.write('\n' + "\nFINAL RESULTS " +
                  '\n' + 'best_params ' + str(best_params) +
                  '\n' + 'best_fscore ' + str(best_fscore) +
                  '\n' + 'best_tp ' + str(best_tp) +
                  '\n' + 'best_fp ' + str(best_fp) +
                  '\n' + 'best_fn ' + str(best_fn))


    # # ************* TEST *************
    # # define the mask
    # m_test = define_mask(len(test_data), test_seqlen, seq_max_len_or_n_steps)
    #
    # acc_test = sess.run(accuracy_test, feed_dict={x: test_data, y: test_labels, seqlen: test_seqlen, mask: m_test, threshold: 0.0})
    # print("Test Accuracy= " + "{:.5f}".format(acc_test))
    #
    # # compute accuracy, precision, recall and fscore for met class
    #
    #
    #
    # thresholded_prediction_test = sess.run(thresholded_pred,feed_dict={x: test_data, y: test_labels, seqlen: test_seqlen, mask: m_test, threshold: 0.0})
    # thresholded_prediction_test_reshaped = tf.reshape(thresholded_prediction_test,[len(test_data), seq_max_len_or_n_steps])
    # thresholded_prediction_test_reshaped_masked = tf.mul(thresholded_prediction_test_reshaped, m_test)
    #
    # # print(thresholded_prediction_test_reshaped_masked.eval())
    #
    # test_labels_reshaped = (tf.reshape(test_labels, [len(test_data), seq_max_len_or_n_steps]))
    #
    # compute_prec_rec_f1_new(thresholded_prediction_test_reshaped_masked, test_labels_reshaped, test_data, seq_max_len_or_n_steps,
    #                         print_tp_etc= True, print_prec=True, print_rec=True, print_f1=True)


