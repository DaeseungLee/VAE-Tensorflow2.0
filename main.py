import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, Model
from model import sampling, Encoder, Decoder, VariationalAutoencoder
from load_data import load_mnist, imshow_mnist

img_dim=784
intermediate_dim=64
latent_dim=16
epochs=40
val_ratio=0.3

def data_preprocessing():
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist(val_ratio)

    x_train = (x_train / 255.0).reshape(-1, img_dim)
    x_val = (x_val / 255.0).reshape(-1, img_dim)
    x_test = (x_test / 255.0).reshape(-1, img_dim)

    return x_train, y_train, x_val, y_val, x_test, y_test

def train():
    x_train, y_train, x_val, y_val, x_test, y_test = data_preprocessing()

    vae = VariationalAutoencoder(img_dim, intermediate_dim, latent_dim)
    x_train = x_train.reshape(-1,img_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.keras.metrics.Mean()

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=256).batch(64)

    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)

                grads = tape.gradient(loss, vae.trainable_weights)
                optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))
        
        reconstructed = vae(x_val)
        loss = mse_loss_fn(x_val, reconstructed)
        loss += sum(vae.losses)
        print("validation loss at epoch %d: %4f" %(step, loss))
        print()


if __name__ == "__main__":
    train()
