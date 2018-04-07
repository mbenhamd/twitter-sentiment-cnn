import ast
import json
import numpy as np
from pyspark import SparkContext
import tensorflow as tf
from data_helpers import load_data, string_to_int, clean_str
import os
import time
import pickle
import copy_reg
import cPickle
import base64
import cloudpickle
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType


def _reduce_method_descriptor(m):
    return getattr, (m.__objclass__, m.__name__)


def weight_variable(shape, name):
    """
    Creates a new Tf weight variable with the given shape and name.
    Returns the new variable.
    """
    var = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(var, name=name)


def bias_variable(shape, name):
    """
    Creates a new Tf bias variable with the given shape and name.
    Returns the new variable.
    """
    var = tf.constant(0.1, shape=shape)
    return tf.Variable(var, name=name)


def evaluate_sentence(sentence):
    x_to_eval = string_to_int(sentence["text_cleaned"], vocabulary, max(len(_) for _ in x))
    result = sess.run(tf.argmax(network_out, 1),
                      feed_dict={data_in: x_to_eval,
                                 dropout_keep_prob: 1.0})
    # unnorm_result = sess.run(network_out, feed_dict={data_in: x_to_eval,
    #                                                 dropout_keep_prob: 1.0})
    network_sentiment = 'POS' if result == 1 else 'NEG'
    # log('Custom input evaluation:', network_sentiment)
    # log('Actual output:', str(unnorm_result[0]))
    if result is not None:	
    	return result[0]
    else:
	return 'ERROR'


def formatter(data):
    return pickle.loads(cloudpickle.dumps(json.dumps(data)))


def transformsentimentTextBloB(data):
    if data["sentiment"] == "negative":
        st = -1
    elif data["sentiment"] == "positive":
        st = 1
    else:
        st = 0
    data["sentiment"] = st
    return data


def transformemotion(data):
    if data["emotion"] == "joy":
        data["emotion"] = 1
    if data["emotion"] == "fear":
        data["emotion"] = 2
    if data["emotion"] == "anger":
        data["emotion"] = 3
    if data["emotion"] == "surprise":
        data["emotion"] = 4
    if data["emotion"] == "sadness":
        data["emotion"] = 5
    return data


def transformsentimentPublication(data):
    if data["sentimentPublication"] == "0":
        data["sentimentPublication"] = -1

    return data


def gospark(json_decoded):
    try:
        json_decoded["sentimentCNN"] = evaluate_sentence(clean_str(json_decoded["text_cleaned"]), vocabulary)
        return json_decoded
    except KeyError:
        return json_decoded


def transform(data):
    line = {
        "itemID": data["itemID"],
        "emotion": data["emotion"],
        "sentimentTextBloB": data["sentiment"],
        "sentimentPublication": data["sentimentPublication"],
        "subjectivityTextBloB": data["subjectivity"],
        "polarityTextBloB": data["polarity"],
        "text_cleaned": clean_str(data["text_cleaned"]),
    }
    return line


def partition_to_encoded_pickle_object(partition):
    p = cPickle.dumps(partition, protocol=2)  # pickle the list
    return pickle.loads([base64.b64encode(p)])  # base64 encode the list, and return it in an iterable


if __name__ == "__main__":
    sc = SparkContext()

    OUT_DIR = os.path.abspath(os.path.join(os.path.curdir, 'output'))
    RUN_ID = time.strftime('run%Y%m%d-%H%M%S')
    RUN_DIR = os.path.abspath(os.path.join(OUT_DIR, RUN_ID))
    LOG_FILE_PATH = os.path.abspath(os.path.join(RUN_DIR, 'log.log'))
    CHECKPOINT_FILE_PATH = os.path.abspath(
        os.path.join("/home/mbenhamd/twitter-sentiment-cnn-master/output/run20180401-174000/", 'ckpt.ckpt'))
    x, y, vocabulary, vocabulary_inv = load_data(1.0)

    np.random.seed(123)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    text_percent = 0.1
    test_index = int(len(x) * text_percent)
    x_train, x_test = x_shuffled[:-test_index], x_shuffled[-test_index:]
    y_train, y_test = y_shuffled[:-test_index], y_shuffled[-test_index:]

    # Parameters
    sequence_length = x_train.shape[1]
    num_classes = y_train.shape[1]
    vocab_size = len(vocabulary)
    filter_sizes = map(int, str("3, 4, 5").split(','))
    validate_every = len(y_train) / (128 * 1)
    checkpoint_every = len(y_train) / (128 * 1)

    # Session
    sess = tf.InteractiveSession()

    # Network
    with tf.device('gpu'):
        # Placeholders
        data_in = tf.placeholder(tf.int32, [None, sequence_length], name='data_in')
        data_out = tf.placeholder(tf.float32, [None, num_classes], name='data_out')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # Stores the accuracy of the model for each batch of the validation testing
        valid_accuracies = tf.placeholder(tf.float32)
        # Stores the loss of the model for each batch of the validation testing
        valid_losses = tf.placeholder(tf.float32)

        # Embedding layer
        with tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, 128],
                                              -1.0, 1.0),
                            name='embedding_matrix')
            embedded_chars = tf.nn.embedding_lookup(W, data_in)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Convolution + ReLU + Pooling layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution Layer
                filter_shape = [filter_size,
                                128,
                                1,
                                128]
                W = weight_variable(filter_shape, name='W_conv')
                b = bias_variable([128], name='b_conv')
                conv = tf.nn.conv2d(embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                # Activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # Maxpooling layer
                ksize = [1,
                         sequence_length - filter_size + 1,
                         1,
                         1]
                pooled = tf.nn.max_pool(h,
                                        ksize=ksize,
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool')
            pooled_outputs.append(pooled)

        # Combine the pooled feature tensors
        num_filters_total = 128 * len(list(filter_sizes))
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Dropout
        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

        # Output layer
        with tf.name_scope('output'):
            W_out = weight_variable([num_filters_total, num_classes], name='W_out')
            b_out = bias_variable([num_classes], name='b_out')
            network_out = tf.nn.softmax(tf.matmul(h_drop, W_out) + b_out)

        # Loss function
        cross_entropy = -tf.reduce_sum(data_out * tf.log(network_out))

        # Training algorithm
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # Testing operations
        correct_prediction = tf.equal(tf.argmax(network_out, 1),
                                      tf.argmax(data_out, 1))
        # Accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Validation ops
        valid_mean_accuracy = tf.reduce_mean(valid_accuracies)
        valid_mean_loss = tf.reduce_mean(valid_losses)

myRDD = sc.textFile("/home/mbenhamd/Downloads/full.json")

analyzed = myRDD.flatMap(gospark)


tmp = analyzed.collect()

tmp.saveAsTextFile("/home/mbenhamd/final")

# saver = tf.train.Saver()
# saver.save(sess, CHECKPOINT_FILE_PATH)
