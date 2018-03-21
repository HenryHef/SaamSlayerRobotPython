import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#from http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

def build_network(ninput,noutput,hidden_layers_sizes=()):
	x = tf.placeholder(tf.float32, [None, ninput])
	# now declare the output data placeholder - 10 digits WANTED OUTPUT
	y = tf.placeholder(tf.float32, [None, noutput])
	
	lsizes = [ninput]
	for i in range(0,len(hidden_layers_sizes)):
		lsizes.append(hidden_layers_sizes[i])
	lsizes.append(noutput)
	
	W,b=[],[]#indexed from 1
	for i in range(1,len(hidden_layers_sizes)+2):
		W.append(tf.Variable(tf.random_normal([lsizes[i-1], lsizes[i]], stddev=0.03), name='W'+str(i)))
		b.append(tf.Variable(tf.random_normal([lsizes[i]]), name='b'+str(i)))
	
	# calculate the output of the hidden layer
	print("o = x*W+b,"+str(W[0])+str(b[0]))
	hidden_out = tf.add(tf.matmul(x, W[0]), b[0])
	hidden_out = tf.nn.relu(hidden_out)
	
	for i in range(1,len(hidden_layers_sizes)):
		print("o = o*W+b,"+str(W[i])+str(b[i]))
		hidden_out = tf.add(tf.matmul(hidden_out, W[i]), b[i])
		hidden_out = tf.nn.relu(hidden_out)
	

	# now calculate the hidden layer output - in this case, let's use a softmax activated
	# output layer
	y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W[-1]), b[-1]))
	print("y_ = o*W+b,"+str(W[-1])+str(b[-1]))
	
	return x,y,y_

def create_vars(dataset):
# Python optimisation variables
	learning_rate = .1
	epochs = 10
	batch_size = 100

	# declare the training data placeholders
	
	#build network
	# input x - for 28 x 28 pixels = 784 INPUT
	x,y,y_ = build_network(28*28,10,(int((28*28+10)/2),))#1 layer, average of the two
	
	# now let's define the cost function which we are going to train the model on
	y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
	cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
												  + (1 - y) * tf.log(1 - y_clipped), axis=1))

	# add an optimiser
	optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

	return (learning_rate,epochs,batch_size,x,y,y_,y_clipped,cross_entropy,optimiser)

def train_network(dataset,vars):
	
	learning_rate,epochs,batch_size,x,y,y_,y_clipped,cross_entropy,optimiser = vars
	
	init_op = tf.global_variables_initializer()

	# define an accuracy assessment operation
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# add a summary to store the accuracy
	tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('/home/henry/workspace/SSE3/neural_chesspiece')
	# start the session
	with tf.Session() as sess:
		# initialise the variables
		sess.run(init_op)
		total_batch = int(len(dataset.train.labels) / batch_size)
		for epoch in range(epochs):
			avg_cost = 0
			for i in range(total_batch):
				batch_x, batch_y = dataset.train.next_batch(batch_size=batch_size)
				_, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
				avg_cost += c / total_batch
			print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
			summary = sess.run(merged, feed_dict={x: dataset.test.images, y: dataset.test.labels})
			writer.add_summary(summary, epoch)

		print("\nTraining complete!")
		writer.add_graph(sess.graph)
		print(sess.run(accuracy, feed_dict={x: dataset.test.images, y: dataset.test.labels}))

def save_sess(sess,path):
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	save_path = saver.save(sess, path)
	print("Model saved in path: %s" % save_path)
	
def restore_sess(sess,path):
	saver = tf.train.Saver()
	saver.restore(sess, path)

if __name__ == "__main__":
	# run_simple_graph()
	# run_simple_graph_multiple()
	# simple_with_tensor_board()

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_network(mnist,create_vars(mnist))
	
 