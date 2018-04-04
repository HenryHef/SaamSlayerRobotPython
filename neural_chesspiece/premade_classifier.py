from data_loader import load_dataset
import numpy as np
import tensorflow as tf


path_train  = "/home/henry/workspace/SSE3/neural_chesspiece/data/"
dataset = load_dataset(path_train,one_hot=False)
dataset=dataset[0]

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=12288)]

# Build 2 layer DNN classifier
print("ABOUT TO classifier")
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[400],
optimizer=tf.train.AdamOptimizer(3e-5),
    n_classes=3,
    dropout=0.1
)

# Define the training inputs
print("ABOUT TO train_input_fn")
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": dataset._data_train},
    y=dataset._labels_train,
    num_epochs=None,
    batch_size=15,
    shuffle=True
)

import logging
logging.getLogger().setLevel(logging.INFO)

print("ABOUT TO TRAIN")
classifier.train(input_fn=train_input_fn, steps=10000)
print("TRAINED")

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": dataset._data_test},
    y=dataset._labels_test,
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

