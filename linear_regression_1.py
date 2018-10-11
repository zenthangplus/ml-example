import os
import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(total_loss)

# Training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Init variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Run training
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

curr_W, curr_b, curr_model, curr_loss = sess.run([W, b, linear_model, total_loss], {x: x_train, y: y_train})
print("W: %s b: %s model: %s, loss: %s" % (curr_W, curr_b, curr_model, curr_loss))
