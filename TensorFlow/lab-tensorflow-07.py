import tensorflow as tf
tf1 = tf.compat.v1
tf1.set_random_seed(777)

g = tf1.Graph()
with g.as_default() as graph:
    x = tf1.placeholder(tf1.float32)
    a = tf1.Variable(tf1.random_normal([1]))
    b = tf1.Variable(tf1.zeros([1]))
    y = a * x + b
    sess = tf1.Session()
    sess.run(tf1.global_variables_initializer())
    print('y = ' + str(sess.run(a)) + 'x + ' + str(sess.run(b)))
    print(sess.run(y, feed_dict={x: [10, 20]}))
