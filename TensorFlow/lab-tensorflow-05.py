import tensorflow as tf
tf1 = tf.compat.v1

g = tf1.Graph()
with g.as_default() as graph:
    x1 = tf1.placeholder(tf1.float32)
    x2 = tf1.placeholder(tf1.float32)
    c = x1 + x2   # c = tf1.add(a, b)
    sess = tf1.Session()
    print(sess.run(c, feed_dict={x1: 3, x2: 4.5}))
    print(sess.run(c, feed_dict={x1: [1, 3], x2: [2, 4]}))
