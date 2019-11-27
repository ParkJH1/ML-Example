import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf1 = tf.compat.v1
tf1.set_random_seed(777)

x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([8, 24, 28, 46, 44, 55, 79, 80, 99, 105])

g = tf1.Graph()
with g.as_default() as graph:
    W = tf1.Variable(tf1.random_normal([1]))
    b = tf1.Variable(tf1.zeros([1]))
    x = tf1.placeholder(tf1.float32)
    y = tf1.placeholder(tf1.float32)

    hypothesis = W * x + b

    cost = tf1.reduce_mean(tf1.square(y - hypothesis))
    optimizer = tf1.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    sess = tf1.Session()
    sess.run(tf1.global_variables_initializer())

    for step in range(1000):
        if step % 100 == 0:
            print('step', step, ': ', sess.run(cost, feed_dict={x: x_data, y: y_data}))
        sess.run(train, feed_dict={x: x_data, y: y_data})

    pred_y = sess.run(hypothesis, feed_dict={x: x_data})
    print(pred_y)

    plt.figure(0)
    plt.plot(x_data, y_data, 'r.', label='data')
    plt.plot(x_data, pred_y, 'b-', label='predict')
    plt.legend()
    plt.show()
