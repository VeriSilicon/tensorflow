"""
the demo which come from:
https://www.jianshu.com/p/6beedc7f83da
"""

import os
# input("pid: " + str(os.getpid()) +", press enter after attached")
import numpy as np
import tensorflow as tf
import json
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_bfloat16')
# mixed_precision.set_policy(policy)

#input("pid: " + str(os.getpid()) +", press enter after set breakpoints")
tf.keras.backend.clear_session()
# tf.config.optimizer.set_jit(True) # Start with XLA disabled.
# tf.debugging.set_log_device_placement(True)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["10.10.161.66:20000", "10.10.161.183:20000"]
    },
    'task': {'type': 'worker', 'index': 0}
})
# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CollectiveCommunication.NCCL))
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

MODEL_FILE = "lenet.json"
MODEL_DATA_FILE = "lenet.h5"

def load_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train[0:19,:,:,:]
y_train = y_train[0:19,:]

x_test = x_test[0:1,:,:,:]
y_test = y_test[0:1,:]

def generate_model():
  with strategy.scope():
    return tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', 
        input_shape=x_train.shape[1:]),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(120, activation='relu'),
      tf.keras.layers.Dense(84, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax', dtype='float32'),
      ])

def compile_model(model):
  opt = tf.keras.optimizers.SGD(lr=0.0001)
  # opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                # run_eagerly=True,
                metrics=['accuracy'])
  return model

log_dir="./logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1,
                                                      update_freq="batch",
                                                      profile_batch=3)

def train_model(model, x_train, y_train, x_test, y_test, epochs=1):
  # model.fit(x_train, y_train, batch_size=20, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[tensorboard_callback])
  model.fit(x_train, y_train, batch_size=10, epochs=epochs, 
    shuffle=True, 
    callbacks=[tensorboard_callback])

def warmup(model, x_train, y_train, x_test, y_test):
  # Warm up the JIT, we do not wish to measure the compilation time.
  initial_weights = model.get_weights()
  train_model(model, x_train, y_train, x_test, y_test, epochs=1)
  model.set_weights(initial_weights)

if os.path.exists(MODEL_FILE):
  json_string = open(MODEL_FILE, 'r').read() 
  model = tf.keras.models.model_from_json(json_string)
  model.load_weights(MODEL_DATA_FILE)
else:
  model = generate_model()

model = compile_model(model)

DUMP_DIR = "npu"

def dump_tensors(tensors, prefix, tensors_name=None):
  if not os.path.isdir(DUMP_DIR): os.makedirs(DUMP_DIR)
  if not os.path.isdir(DUMP_DIR + "/" + prefix):
    os.makedirs(DUMP_DIR + "/" + prefix)
  if (tensors_name == None):
    ts_zip = zip(tensors, tensors)
  else:
    ts_zip = zip(tensors, tensors_name)
  for tensor in ts_zip:
    file_name = DUMP_DIR + "/" + prefix + "/" + \
        tensor[1].name.replace("/", "_").replace(":", "_") + ".txt"
    print(tensor[1].name, tensor[1].shape, " saved in: ", file_name)
    if hasattr(tensor[0], "numpy"):
      num = tensor[0].numpy()
    else:
      num = tensor[0]
    np.savetxt(file_name, num.flatten(), fmt='%.8f')

# weights = [weight for layer in model.layers for weight in layer.weights]
# dump_tensors(weights, "before")

def get_weight_grad(model, inputs, outputs):
  with tf.GradientTape() as tape:
    pred = model(inputs)
    loss = model.compiled_loss(tf.convert_to_tensor(outputs), pred, None,
                                   regularization_losses=model.losses)
  grad = tape.gradient(loss, model.trainable_weights)
  return grad

#warmup(model, x_train, y_train, x_test, y_test)
train_model(model, x_train, y_train, x_test, y_test, epochs=1)

print("RRR : job finish.")
