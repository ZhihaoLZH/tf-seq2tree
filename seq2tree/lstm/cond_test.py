import tensorflow as tf

a = tf.constant(1)
b = tf.constant(1)

a2 = tf.constant(2)
b2 = tf.constant(2)

def true_fn(x):
    # str = tf.constant(['true_fn'])
    # str = tf.Print(str, [str])
    x = tf.Print(x, [x])

    return x

def false_fn(x):
    # str = 'str'
    # str = tf.Print(str, [str])
    # x = tf.Print(x, [x])

    return x

c = tf.cond(tf.logical_and(tf.equal(a, b), tf.equal(a2, b2)), true_fn=lambda: true_fn(a), false_fn=lambda: false_fn(a2))

with tf.Session() as sess:
     sess.run(c)
