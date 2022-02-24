import os
input("pid: " + str(os.getpid()) +", press enter after attached")
import numpy as np
import tensorflow as tf
print("tf verison: " + tf.__version__)
# input("pid: " + str(os.getpid()) +", press enter after set breakpoints")
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Start with XLA disabled.
tf.debugging.set_log_device_placement(True)

MODEL_FILE = "resnet.json"
MODEL_DATA_FILE = "resnet.h5"

BATCH_SIZE = 2
TRAIN_SIZE = BATCH_SIZE
EPOCHS = 1
DUMP_DIR = "npu"
MODEL_NAME = "tiny_v1_for_full_size"

def load_cifar_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / 256
  x_test = x_test.astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

# fake data
def load_fake_data(train_size, test_size, height, width, channel, num_classes):
  rng = np.random.default_rng(12345)
  x_train = rng.random((train_size, height, width, channel))
  x_test = rng.random((test_size, height, width, channel))

  y_train = rng.integers(low=0, high=1000, size=(train_size,1))
  y_test = rng.integers(low=0, high=1000, size=(test_size,1))
  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
  return ((x_train, y_train), (x_test, y_test))

def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
  x = tf.keras.layers.Conv2D(
    filters, kernel_size, strides=strides, padding='same')(input_layer)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  return x

def resiidual_a_or_b(input_x, filters, flag):
  if flag == 'a':
    x = Conv_BN_Relu(filters, (3, 3), 1, input_x)
    x = Conv_BN_Relu(filters, (3, 3), 1, x)
    y = tf.keras.layers.Add()([x, input_x])
    return y
  elif flag == 'b':
    x = Conv_BN_Relu(filters, (3, 3), 1, input_x)
    x = Conv_BN_Relu(filters, (3, 3), 1, x)

    input_x = Conv_BN_Relu(filters, (1, 1), 1, input_x)
    y = tf.keras.layers.Add()([x, input_x])
    return y

"""
this resnetv1 demo which come from:
https://blog.csdn.net/my_name_is_learn/article/details/109640715
"""
def generate_v1_model():
  input_layer = tf.keras.layers.Input((224, 224, 3))
  conv1 = Conv_BN_Relu(64, (7, 7), 1, input_layer)
  conv1_Maxpooling = tf.keras.layers.MaxPooling2D((3, 3),
    strides=2, padding='same')(conv1)
  # conv2_x
  x = resiidual_a_or_b(conv1_Maxpooling, 64, 'b')
  x = resiidual_a_or_b(x, 64, 'a')
  # conv3_x
  x = resiidual_a_or_b(x, 128, 'b')
  x = resiidual_a_or_b(x, 128, 'a')
  # conv4_x
  x = resiidual_a_or_b(x, 256, 'b')
  x = resiidual_a_or_b(x, 256, 'a')
  # conv5_x
  x = resiidual_a_or_b(x, 512, 'b')
  x = resiidual_a_or_b(x, 512, 'a')
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(1000)(x)
  # x = tf.keras.layers.Dropout(0.5)(x)
  y = tf.keras.layers.Softmax(axis=-1)(x)
  model = tf.keras.models.Model([input_layer], [y])
  return model

# modify for cifar dataset
def generate_cifar_model():
  input_layer = tf.keras.layers.Input((32, 32, 3))
  conv1 = Conv_BN_Relu(64, (3, 3), 1, input_layer)
  x = resiidual_a_or_b(conv1, 64, 'b')
  x = resiidual_a_or_b(x, 64, 'a')
  # conv3_x
  x = resiidual_a_or_b(x, 128, 'b')
  x = resiidual_a_or_b(x, 128, 'a')
  # conv4_x
  x = resiidual_a_or_b(x, 256, 'b')
  x = resiidual_a_or_b(x, 256, 'a')
  # conv5_x
  x = resiidual_a_or_b(x, 512, 'b')
  x = resiidual_a_or_b(x, 512, 'a')
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(10)(x)
  y = tf.keras.layers.Softmax(axis=-1)(x)
  model = tf.keras.models.Model([input_layer], [y])
  return model

def generate_tiny_model():
  input_layer = tf.keras.layers.Input((224, 224, 3))
  conv1 = Conv_BN_Relu(64, (3, 3), 1, input_layer)
  x = resiidual_a_or_b(conv1, 64, 'b')
  x = resiidual_a_or_b(x, 64, 'a')

  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(1000)(x)
  y = tf.keras.layers.Softmax(axis=-1)(x)
  model = tf.keras.models.Model([input_layer], [y])
  return model

# modify for cifar dataset
def generate_tiny_cifar_model():
  input_layer = tf.keras.layers.Input((32, 32, 3))
  conv1 = Conv_BN_Relu(64, (3, 3), 1, input_layer)
  x = resiidual_a_or_b(conv1, 64, 'b')
  x = resiidual_a_or_b(x, 64, 'a')

  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(10)(x)
  y = tf.keras.layers.Softmax(axis=-1)(x)
  model = tf.keras.models.Model([input_layer], [y])
  return model

def generate_v2_model():
  model = tf.keras.applications.ResNet50V2()
  return model

def compile_model(model):
  opt = tf.keras.optimizers.SGD(lr=0.0001)
  # opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                # run_eagerly=True,
                metrics=['accuracy'])
  return model

def train_model(model, x_train, y_train, x_test,
  y_test, epochs=1, batch_size=1):
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
    validation_data=(x_test, y_test), shuffle=False)

def warmup(model, x_train, y_train, x_test, y_test):
  # Warm up the JIT, we do not wish to measure the compilation time.
  initial_weights = model.get_weights()
  train_model(model, x_train, y_train, x_test, y_test, epochs=1)
  model.set_weights(initial_weights)

def get_model(model_list, model_name):
  for model in model_list:
    if (model['name'] == model_name):
      return model

model_list = [
    {'name': "offical_v2",
     'model': 'generate_v2_model()',
     'data_set': 'load_fake_data(TRAIN_SIZE, 1, 224, 224, 3, 1000)',
    },
    {'name': "unoffical_v1",
     'model': 'generate_v1_model()',
     'data_set': 'load_fake_data(TRAIN_SIZE, 1, 224, 224, 3, 1000)',
    },
    {'name': "unoffical_v1_for_cifar",
     'model': 'generate_cifar_model()',
     'data_set': 'load_cifar_data()',
    },
    {'name': "tiny_v1_for_cifar",
     'model': 'generate_tiny_cifar_model()',
     'data_set': 'load_cifar_data()',
    },
    {'name': "tiny_v1_for_full_size",
     'model': 'generate_tiny_model()',
     'data_set': 'load_fake_data(TRAIN_SIZE, 1, 224, 224, 3, 1000)',
    },
]

model_info = get_model(model_list, MODEL_NAME)
print (model_info)

(x_train, y_train), (x_test, y_test) = eval(model_info['data_set'])

x_train = x_train[0:TRAIN_SIZE, :, :, :]
y_train = y_train[0:TRAIN_SIZE, :]

x_test = x_test[0:1,:,:,:]
y_test = y_test[0:1,:]

if os.path.exists(MODEL_FILE):
  json_string = open(MODEL_FILE, 'r').read() 
  model = tf.keras.models.model_from_json(json_string)
  model.load_weights(MODEL_DATA_FILE)
else:
  model = eval(model_info['model'])

model = compile_model(model)

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

weights = [weight for layer in model.layers for weight in layer.weights]
dump_tensors(weights, "before")

def get_weight_grad(model, inputs, outputs):
  with tf.GradientTape() as tape:
    pred = model(inputs)
    loss = model.compiled_loss(tf.convert_to_tensor(outputs), pred, None,
                                   regularization_losses=model.losses)
  grad = tape.gradient(loss, model.trainable_weights)
  return grad

#warmup(model, x_train, y_train, x_test, y_test)
train_model(model, x_train, y_train, x_test, y_test,
  epochs=EPOCHS, batch_size=BATCH_SIZE)
model.summary()

weight_grads = get_weight_grad(model, x_train, y_train)
dump_tensors(weight_grads, "grads", model.trainable_weights)

if not os.path.exists(MODEL_FILE):
  json_string = model.to_json()
  open(MODEL_FILE, 'w').write(json_string) 
  model.save_weights(MODEL_DATA_FILE)
  print("RRR : save model.")

dump_tensors(weights, "after")
model.save_weights(DUMP_DIR + "/after/" + MODEL_DATA_FILE)

# outputs = [layer.output for layer in model.layers][1:]
# # all layer outputs except first (input) layer
# functor = tf.keras.backend.function([model.input], outputs)
# layer_outs = functor([x_train])
# dump_tensors(layer_outs, "output", outputs)

# print(model.predict(x_test))

print("RRR : job finish.")
