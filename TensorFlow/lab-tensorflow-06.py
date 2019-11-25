import tensorflow as tf
tf1 = tf.compat.v1
tf1.set_random_seed(777)

g = tf1.Graph()
with g.as_default() as graph:
    a = tf1.Variable(tf1.random_normal([10]))
    b = tf1.Variable(tf1.zeros([10]))
    sess = tf1.Session()
    sess.run(tf1.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
