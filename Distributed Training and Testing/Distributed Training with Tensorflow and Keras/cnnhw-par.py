#Group Author info:
#psomash Prakruthi Somashekarappa
#rbraman Radhika B Raman
#srames22 Srivatsan Ramesh

# -*- coding: utf-8 -*-
## Stage 1: Installing dependencies and notebook gpu setup
import os
import sys
import json
import time

#used this to record time taken by parallel version of the code
#start_time = time.time()

#!pip install tensorflow-gpu==2.0.0.alpha0

# handling one parameter on the command line to indicate whether the program is running on evaluator/worker
if len(sys.argv) != 2:
    print ("Usage:", sys.argv[0], "-1 for evaluator OR N for workers")
    sys.exit()
    
cmd_input = int(sys.argv[1])

## Stage 2: Importing dependencies for the project

os.environ['NCCL_P2P_DISABLE'] = "1"
# Commented out IPython magic to ensure Python compatibility.
#get a local copy of datasets
os.system("ln -s /mnt/beegfs/fmuelle/.keras ~/")
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.datasets import cifar10

#for RTX GPUs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# %matplotlib inline
tf.__version__

# setting TF_CONFIG environment variable
if(cmd_input != -1):
    os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["c103:8000", "c104:8001"]
    },
    'task': {'type': 'worker', 'index': cmd_input}
    })

# Create a MirroredStrategy.
if(cmd_input == -1):
    strategy = tf.distribute.MirroredStrategy()

# Create a MultiWorkerMirroredStrategy
else:
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# setting cdir/tdir pathnames as per instructions

# Prepare a directory to store all the checkpoints.
checkpoint_dir1_eval_chief = "/home/rbraman/ckpt"
checkpoint_dir2_eval_chief = "/home/rbraman/tb"

checkpoint_dir1_other_workers = "/tmp/rbraman/ckpt"
checkpoint_dir2_other_workers = "/tmp/rbraman/tb"

if(cmd_input == 0 or cmd_input == -1):
    if os.path.exists(checkpoint_dir1_eval_chief):
        os.system("rm -rf " + checkpoint_dir1_eval_chief)
        
    if os.path.exists(checkpoint_dir2_eval_chief):
        os.system("rm -rf " + checkpoint_dir2_eval_chief)

    os.makedirs(checkpoint_dir1_eval_chief)
    os.makedirs(checkpoint_dir2_eval_chief)

else:
    if os.path.exists(checkpoint_dir1_other_workers):
        os.system("rm -rf " + checkpoint_dir1_other_workers)
        
    if os.path.exists(checkpoint_dir2_other_workers):
        os.system("rm -rf " + checkpoint_dir2_other_workers)

    os.makedirs(checkpoint_dir1_other_workers)
    os.makedirs(checkpoint_dir2_other_workers)


## Stage 3: Dataset preprocessing

### Loading the Cifar10 dataset

def get_dataset():
    batch_size = 256
    #num_val_samples = 50000

    #Setting class names for the dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #Loading the dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    ### Image normalization

    X_train = X_train.astype("float32") / 255.0

    X_train.shape

    X_test = X_test.astype("float32") / 255.0

    plt.imshow(X_test[10])

    y_train = y_train.astype("float32")

    y_test = y_test.astype("float32")

    return (
        tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    )


## Stage 4: Building a Convolutional neural network

### Defining the model
def get_compiled_model():

    model = tf.keras.models.Sequential()

    ### Adding the first CNN Layer

    #CNN layer hyper-parameters:
    #- filters: 32
    #- kernel_size:3
    #- padding: same
    #- activation: relu
    #- input_shape: (32, 32, 3)



    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))

    ### Adding the second CNN Layer and max pool layer

    #CNN layer hyper-parameters:
    #- filters: 32
    #- kernel_size:3
    #- padding: same
    #- activation: relu

    #MaxPool layer hyper-parameters:
    #- pool_size: 2
    #- strides: 2
    #- padding: valid


    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    ### Adding the third CNN Layer

    #CNN layer hyper-parameters:

    #    filters: 64
    #    kernel_size:3
    #    padding: same
    #    activation: relu
    #    input_shape: (32, 32, 3)



    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

    ###  Adding the fourth CNN Layer and max pool layer

    #CNN layer hyper-parameters:

    #    filters: 64
    #    kernel_size:3
    #    padding: same
    #    activation: relu

    #MaxPool layer hyper-parameters:

    #    pool_size: 2
    #    strides: 2
    #    padding: valid



    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    ### Adding the Flatten layer

    model.add(tf.keras.layers.Flatten())

    ### Adding the first Dense layer

    #Dense layer hyper-parameters:
    #- units/neurons: 128
    #- activation: relu


    model.add(tf.keras.layers.Dense(units=128, activation='relu'))

    ### Adding the second Dense layer (output layer)

    #Dense layer hyper-parameters:

    # - units/neurons: 10 (number of classes)
    # - activation: softmax



    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.summary()

    ### Compiling the model

    #### sparse_categorical_accuracy
    #sparse_categorical_accuracy checks to see if the maximal true value is equal to the index of the maximal predicted value.

    #https://stackoverflow.com/questions/44477489/keras-difference-between-categorical-accuracy-and-sparse-categorical-accuracy 


    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="Adam", metrics=["sparse_categorical_accuracy"])

    return model

checkpoints=[]

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.

    if(cmd_input == 0 or cmd_input == -1):
        checkpoints = [checkpoint_dir1_eval_chief + "/" + name for name in os.listdir(checkpoint_dir1_eval_chief)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            try:
                return keras.models.load_model(latest_checkpoint)
            except OSError:
                time.sleep(1)
                print("Exception handled")
                return keras.models.load_model(latest_checkpoint)
    else:
        checkpoints = [checkpoint_dir1_other_workers + "/" + name for name in os.listdir(checkpoint_dir1_other_workers)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            return keras.models.load_model(latest_checkpoint)
        else:
            time.sleep(1)
            print(".", end='')

    print("Creating a new model")
    return get_compiled_model()


def run_training(epochs=1):

    # Open a strategy scope and create/restore the model
    if(cmd_input == -1):
        prev_len=0
        length=len(os.listdir(checkpoint_dir1_eval_chief))
        while(length < epochs):
            while(prev_len == length):
                time.sleep(1)
                print(".", end='')
                length=len(os.listdir(checkpoint_dir1_eval_chief))
            
            prev_len = length
            with strategy.scope():
                model = make_or_restore_model()
            #evaluate model for test data
            test_loss, test_accuracy = model.evaluate(test_dataset)
            length=len(os.listdir(checkpoint_dir1_eval_chief))

            print("Test accuracy: {}".format(test_accuracy))
            print("Epoch number: {}".format(length))

    else:
        with strategy.scope():
            model = make_or_restore_model()

    
    if(cmd_input != -1):

        if(cmd_input == 0):

            callbacks = [
                keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir1_eval_chief + "/ckpt-{epoch}", save_freq="epoch"),
                keras.callbacks.TensorBoard(checkpoint_dir2_eval_chief + "/")
            ]
        else:
            callbacks = [
                keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir1_other_workers + "/ckpt-{epoch}", save_freq="epoch"),
                keras.callbacks.TensorBoard(checkpoint_dir2_other_workers + "/")
            ]
        ### Training the model
        model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=callbacks
        )

    # else:
    #     #evaluate model for test data
    #     test_loss, test_accuracy = model.evaluate(test_dataset)

    #     print("Test accuracy: {}".format(test_accuracy))
    #     print("Epoch number: {}".format(epochs))

train_dataset, test_dataset = get_dataset()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_dataset = train_dataset.with_options(options)
run_training(epochs=6)
#print("--- %s seconds ---" % (time.time() - start_time))


