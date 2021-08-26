#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
input("pid: " + str(os.getpid()) +", AAA press enter after attached")
import tensorflow as tf

from tensorflow.python.client import device_lib
import numpy as np
#from tensorflow.profiler.experimental import Profile

input("pid: " + str(os.getpid()) +", press enter after attached")
input("pid: " + str(os.getpid()) +", press enter after set breakpoints")

#print(device_lib.list_local_devices())
fit_flag = True

def test1():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  x_train = np.float32(x_train)
  x_test = np.float32(x_test)
  x_train, x_test = np.reshape(x_train, (-1, 28, 28, 1)), np.reshape(x_test, (-1, 28, 28, 1))

  with tf.device("/device:XLA_CPU:0"):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(1, 3, input_shape=(28, 28, 1)),
    ])

  true_result = np.ones((1,26,26,1),dtype=np.float32)
  predictions = model([x_train[:1],x_train[:1]]).numpy()

  loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  loss_fn(true_result, predictions).numpy()

  model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

  # if fit_flag:
  #   model.fit(x_train, y_train, epochs=1)

  #   model.evaluate(x_test,  y_test, verbose=2)

  #   probability_model = tf.keras.Sequential([
  #     model,
  #     tf.keras.layers.Softmax()
  #   ])

  #   probability_model(x_test[:5])

def test2():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  with tf.device("/device:XLA_CPU:0"):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()

    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    if fit_flag:
      model.fit(x_train, y_train, epochs=1)

      model.evaluate(x_test,  y_test, verbose=2)

      probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
      ])

      probability_model(x_test[:5])

@tf.function
def train_one_step(X, Y):
  #with tf.xla.experimental.jit_scope(compile_ops=True,separate_compiled_gradients=True):
  with tf.xla.experimental.jit_scope(compile_ops=True):
    t1 = tf.add(X, Y)
    return t1

def test3():
  a = np.float32([[1,2,3],[4,5,6]])
  b = np.float32([[4,5,6],[7,8,9]])

  res = train_one_step(a, b)
  print(res.numpy())

#test()
test1()
#test2()
#test3()