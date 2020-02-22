import pickle
import tensorflow as tf

train_test_size = 10000

input_list = pickle.load(open("list.p", "rb"))
#input_list =input_list[:train_test_size]

print(type(input_list))
print(len(input_list))

print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

print("done")
