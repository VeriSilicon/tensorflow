# Import TensorFlow
import tensorflow as tf
import numpy as np
import json
import os

print(tf.__version__)

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["10.10.161.66:20000", "10.10.161.183:20000"]
    },
    'task': {'type': 'worker', 'index': 0}
})


# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(), devices=["/CPU:0", "/CPU:1"])
# strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(), devices=["/GPU:0"])
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CollectiveCommunication.NCCL))
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

train_dataset = tf.data.Dataset.from_tensors(
    (np.zeros(shape=(28, 28, 1), dtype=np.float32), [1.])).repeat(20).batch(20)
# test_dataset = tf.data.Dataset.from_tensors((np.zeros(shape=(28,28,1), dtype=np.float32), [1.])).repeat(200).batch(10)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #   tf.keras.layers.MaxPooling2D(),
    #   tf.keras.layers.Conv2D(64, 3, activation='relu'),
    #   tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
    #   tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
model.compile(loss='mse', optimizer='sgd')

log_dir="./logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1,
                                                      update_freq="batch",
                                                      profile_batch='1,10')
model.fit(train_dataset, epochs=1, callbacks=[tensorboard_callback])
