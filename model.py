# -*- coding: utf-8 -*-

# for mounting with goggle drives
# import sys
# sys.path.append('/content/drive/MyDrive/tesorflow2.0 연습/VAE')
# print(sys.path)

import load_data
import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Conv2DTranspose, Reshape, Input


def sampling(mean, sigma):
  shape = np.shape(mean.numpy())
  return tf.random.normal(shape, mean, sigma, dtype=tf.float32)

class Encoder(layers.Layer):
  def __init__(self,
               latent_dim=16,
               intermediate_dim=64,
               name='encoder'
               ):
    super(Encoder, self).__init__(name=name)
    
    self.h0 = Dense(intermediate_dim, activation='relu')
    self.h1 = Dense(intermediate_dim, activation='relu')
    self.mean = Dense(latent_dim)
    self.sigma = Dense(latent_dim)

  def call(self, inputs):
    h0 = self.h0(inputs)
    h1 = self.h1(h0)
    mean = self.mean(h1)
    sigma = self.sigma(h1)
    z = sampling(mean, sigma)
    return mean, sigma, z

class Decoder(layers.Layer):
  def __init__(
      self,
      intermediate_dim = 64,
      img_dim=784,
      name='decoder'
  ):
    super(Decoder, self).__init__(name=name)
    self.h0 = Dense(intermediate_dim, activation='relu')
    self.h1 = Dense(intermediate_dim, activation='relu')
    self.reconstruct = Dense(img_dim, activation='sigmoid')

  def call(self, inputs):
    h0 = self.h0(inputs)
    h1 = self.h1(h0)
    reconstruct = self.reconstruct(h1)
    return reconstruct

class VariationalAutoencoder(layers.Layer):
  def __init__(self,
               img_dim,
               intermediate_dim,
               latent_dim,
               name='vae'
               ):
    super(VariationalAutoencoder, self).__init__(name=name)
    self.encoder=Encoder(latent_dim=latent_dim,
                         intermediate_dim=intermediate_dim,
                         )
    self.decoder=Decoder(intermediate_dim=intermediate_dim,
                         img_dim=img_dim)
    
  
  def call(self, inputs):
    z_mean, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    kl_loss = - 0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    self.add_loss(kl_loss)
    return reconstructed



