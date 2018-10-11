import tensorflow as tf

# Init session
sess = tf.Session()

# Declare variables
W = tf.Variable(0, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32)

# Create model
linear_model = W*x + b

# Init variables
sess.run(tf.global_variables_initializer())

# Assign new value
sess.run([tf.assign(W, -1), tf.assign(b, 1)])

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
