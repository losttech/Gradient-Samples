import math
import numpy as np
import tensorflow as tf

x0, x1 = 10, 20 # data range

test_data_size = 2000
iterations = 1000 # training iterations
learn_rate = 0.01

hiddenSize = 10

def generate_test_values():
    train_x = []
    train_y = []

    for _ in range(test_data_size):
        x = x0+(x1-x0)*np.random.rand()
        y = math.sin(x) # approximating this
        train_x.append([x])
        train_y.append([y])

    return np.array(train_x), np.array(train_y)

input = tf.placeholder(tf.float32, [None, 1], name="x")

# placeholder for known outputs
output = tf.placeholder(tf.float32, [None, 1], name="y")

# hidden layer
nn = tf.layers.dense(input, hiddenSize,
                     activation=tf.nn.sigmoid,
                     kernel_initializer=tf.initializers.ones(),
                     bias_initializer=tf.initializers.random_uniform(minval=-x1, maxval=-x0),
                     name="hidden")

# output layer
model = tf.layers.dense(nn, 1,
                        activation=None,
                        name="output")

# error cost
cost = tf.losses.mean_squared_error(output, model)

train = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

init = tf.initializers.global_variables()

with tf.Session() as session:
    session.run(init)

    for _ in range(iterations):

        train_dataset, train_values = generate_test_values()

        session.run(train, feed_dict={
            input: train_dataset,
            output: train_values
        })

        if(_ % 100 == 99):
            print("cost = {}".format(session.run(cost, feed_dict={
                input: train_dataset,
                output: train_values
            })))

    train_dataset, train_values = generate_test_values()

    train_values1 = session.run(model, feed_dict={
        input: train_dataset,
    })

    with tf.variable_scope("hidden", reuse=True):
        w = tf.get_variable("kernel")
        b = tf.get_variable("bias")
        print("hidden:")
        print("kernel=", w.eval())
        print("bias = ", b.eval())

    with tf.variable_scope("output", reuse=True):
        w = tf.get_variable("kernel")
        b = tf.get_variable("bias")
        print("output:")
        print("kernel=", w.eval())
        print("bias = ", b.eval())