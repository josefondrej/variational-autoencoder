import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt
import random

obs_dim = 28**2
latent_dim = 16
batch_size = 128
L = 16  # how many samples to generate from q_{\phi}(z | x) to
        # estimate E_{q_{\phi}(z | x)} [ log p_{\theta}(x | z) ]

# Build the graph

x = tf.placeholder(dtype=tf.float32, shape=(batch_size, obs_dim))
eps = tf.random.normal([batch_size, L, latent_dim], 0, 1)

# Encoder -- q_{\phi}(z | x)
mu = tf.layers.dense(x, 32, activation=tf.nn.relu)
mu = tf.layers.dense(mu, latent_dim)

sigma = tf.layers.dense(x, 32, activation=tf.nn.relu)
sigma = tf.exp(tf.layers.dense(sigma, latent_dim))

# D_KL
D_KL = -0.5 * (latent_dim +
              tf.reduce_sum(tf.log(tf.square(sigma)), axis=-1) -
              tf.reduce_sum(tf.square(sigma), axis=-1) -
              tf.reduce_sum(tf.square(mu), axis=-1))

# Sampling Z
z_sample = tf.tile(tf.expand_dims(mu, axis = 1), [1, L, 1]) + tf.tile(tf.expand_dims(sigma, axis = 1), [1, L, 1]) * eps

# Decoder
probas = tf.layers.dense(z_sample, 32, activation=tf.nn.relu)
probas = tf.layers.dense(probas, obs_dim, activation=tf.nn.sigmoid)

log_lik = tf.log(probas) * tf.tile(tf.expand_dims(x, axis = 1), [1, L, 1]) + tf.log(1-probas) * (1-tf.tile(tf.expand_dims(x, axis = 1), [1, L, 1]))

log_lik = tf.reduce_mean(tf.reduce_mean(log_lik, axis=2), axis = 1)

loss = tf.reduce_mean(log_lik - D_KL)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
step = optimizer.minimize(-loss)

## Train the model

# Load the data
(x_train, _),(_, _) = mnist.load_data()
x_train = 1.0*(x_train.reshape(-1,28**2)>128)
plt.imshow(x_train[10].reshape(28,28))


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(1000):
    x_feed = x_train[np.random.randint(x_train.shape[0], size=batch_size)]
    loss_eval, step_eval = sess.run([loss, step], feed_dict = {x: x_feed})
    if i % 100 == 0 and i > 1:
        print(f"Step: {i}, loss: {loss_eval}")
        image = sess.run(probas, feed_dict={z_sample: np.random.randn(batch_size, L, latent_dim)})[0][0].reshape(28,28)
        plt.imshow(image)
        plt.show()




