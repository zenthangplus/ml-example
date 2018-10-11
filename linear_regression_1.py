import os
import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Init session
sess = tf.Session()

# Declare variables
W = tf.Variable(0, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32)

# Create model
linear_model = W*x + b

# Loss function
y = tf.placeholder(tf.float32)
loss = tf.square(linear_model - y)
total_loss = tf.reduce_sum(loss)

# Init variables
sess.run(tf.global_variables_initializer())

# Assign new value
new_w = tf.assign(W, -1)
new_b = tf.assign(b, 1)
sess.run([new_w, new_b])

# Training data
source = [1, 2, 3, 4]
dist = [0, -1, -2, -3]

# Run model
print("Exec: ", str(sess.run(linear_model, {x: source})))

# Calc loss
print("Loss: ", float(sess.run(total_loss, {x: source, y: dist})))
