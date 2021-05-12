import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import datasets

val_ratio = 0.8

def load_mnist(val_ratio):
  (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
  
  # data noramlize
  x_train = x_train / 255.0
  x_test = x_test / 255.0

  train_length = int(len(x_train)*val_ratio)
  
  x_val = x_train[train_length:]
  y_val = y_train[train_length:]

  x_train = x_train[:train_length]
  y_train = y_train[:train_length]

  return x_train, y_train, x_val, y_val, x_test, y_test

def imshow_mnist(x_train, n=10):
  plt.figure(figsize=(20,4))
  for i in range(n):
    ax = plt.subplot(1,n,i+1)
    plt.imshow(x_train[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

