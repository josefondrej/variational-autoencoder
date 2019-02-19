import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------------------------
# Parameters

obs_dim = 28**2
latent_dim = 2
batch_size = 64
L = 16  # how many samples to generate from q_{\phi}(z | x) to
        # estimate E_{q_{\phi}(z | x)} [ log p_{\theta}(x | z) ]

# ------------------------------------------------------------------------------
# Build the graph

x = tf.placeholder(dtype=tf.float32, shape=(batch_size, obs_dim))
eps = tf.random.normal([batch_size, L, latent_dim], 0, 1)

# Encoder -- q_{\phi}(z | x)
h = tf.layers.dense(x, 32, activation=tf.nn.relu)

mu = tf.layers.dense(h, 16, activation=tf.nn.relu)
mu = tf.layers.dense(mu, latent_dim)

sigma = tf.layers.dense(h, 16, activation=tf.nn.relu)
sigma = tf.exp(tf.layers.dense(sigma, latent_dim))

# D_KL
D_KL = -0.5 * (latent_dim +
              tf.reduce_sum(tf.log(tf.square(sigma)), axis=-1) -
              tf.reduce_sum(tf.square(sigma), axis=-1) -
              tf.reduce_sum(tf.square(mu), axis=-1))

# Sampling Z
z_sample = tf.tile(tf.expand_dims(mu, axis = 1), [1, L, 1]) + tf.tile(tf.expand_dims(sigma, axis = 1), [1, L, 1]) * eps

# Decoder -- p_{\theta}(x | z)
probas = tf.layers.dense(z_sample, 32, activation=tf.nn.relu)
probas = tf.layers.dense(probas, obs_dim, activation=tf.nn.sigmoid)

log_lik = tf.log(probas) * tf.tile(tf.expand_dims(x, axis = 1), [1, L, 1]) + tf.log(1-probas) * (1-tf.tile(tf.expand_dims(x, axis = 1), [1, L, 1]))

log_lik = tf.reduce_mean(tf.reduce_mean(log_lik, axis=2), axis = 1)

loss = tf.reduce_mean(log_lik - D_KL)
# loss = tf.reduce_mean(log_lik)

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
step = optimizer.minimize(-loss)

saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Train the model

# Load the data
(x_train, _),(_, _) = mnist.load_data()
x_train = 1.0*(x_train.reshape(-1,28**2)>128)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# iterations = 60000 * 10
# print_freq = 10000

iterations = 10000
print_freq = 1000

for i in range(iterations):
    x_feed = x_train[np.random.randint(x_train.shape[0], size=batch_size)]
    loss_eval, step_eval = sess.run([loss, step], feed_dict = {x: x_feed})
    if i % print_freq == 0 and i > 1:
        print(f"Step: {i}, loss: {loss_eval}")
        # image = sess.run(probas, feed_dict={z_sample: np.tile(np.array([[[0.,0.]]]), [batch_size, L, 1])})[0][0].reshape(28,28)

save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in path: %s" % save_path)

# ------------------------------------------------------------------------------
# Evaluation part

z_sample_realization = [0.0, 0.0]
z_sample_realization = np.array([[z_sample_realization for i in range(L)] for j in range(batch_size)]) # just a dirty trick so we can use the existing architecture
probas_realization = sess.run(probas, feed_dict={z_sample: z_sample_realization})
probas_realization = probas_realization[0][0]
probas_img = probas_realization.reshape(28, 28)
plt.imshow(probas_img)
