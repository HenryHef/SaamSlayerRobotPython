import tensorflow as tf
from data_loader import load_dataset
import time
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

def create_vars(i_size,h_layers,o_size,epochs = 10,batch_size = 100,):
# Python optimisation variables
	

	learning_rate = tf.placeholder(tf.float32, shape=[])
	# declare the training data placeholders
	
	#build network
	# input x - for 28 x 28 pixels = 784 INPUT
	x,y,y_ = build_network(i_size,o_size,h_layers)#1 layer, average of the two
	
	# now let's define the cost function which we are going to train the model on
	y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
	cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
												  + (1 - y) * tf.log(1 - y_clipped), axis=1))

	# add an optimiser
	optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

	return (learning_rate,epochs,batch_size,x,y,y_,y_clipped,cross_entropy,optimiser)

def train_network(dataset,vars,lrate = .03, learning_rate_slow_param = 500.0):
	
	learning_rate,epochs,batch_size,x,y,y_,y_clipped,cross_entropy,optimiser = vars
	
	init_op = tf.global_variables_initializer()

	# define an accuracy assessment operation
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# add a summary to store the accuracy
	tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()
	print("starting")
	# start the session
	with tf.Session() as sess:
		print("started")
		# initialise the variables
		sess.run(init_op)
		print("starting epoch 0")
		data_test,labels_test = dataset.getTestData()
		print("Accuracy:",sess.run(accuracy, feed_dict={x: data_test, y: labels_test}))
		for epoch in range(epochs):
			learning_rate_r = lrate/(1.0+epoch/learning_rate_slow_param)
			
			dataset.newEpoch()
			avg_cost = 0
			indx = 0
			while(dataset.isNextBatch()):
				batch_x, batch_y, batch_ct = dataset.getNextBatch(batch_size)
# 				print(batch_x.shape,batch_y.shape)
				indx+=batch_size
				trash, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y, learning_rate: learning_rate_r})
				avg_cost += c * batch_ct / dataset.trainCt()
# 				print("cost =", "{:.3f}".format(avg_cost))
			print("Epoch:", (epoch + 1), " cost =", "{:.5f}".format(avg_cost),"  lrate=","{:.5f}".format(learning_rate_r))
			data_test,labels_test = dataset.getTestData()
			summary = sess.run(merged, feed_dict={x: data_test, y: labels_test})
			if(epoch%10==0):
				data_test,labels_test = dataset.getTestData()
				print("Accuracy:",sess.run(accuracy, feed_dict={x: data_test, y: labels_test}))
		
		print("\nTraining complete!")
		data_test,labels_test = dataset.getTestData()
		print(sess.run(accuracy, feed_dict={x: data_test, y: labels_test}))
		save_sess(sess,"/home/henry/workspace/SSE3/neural_chesspiece/nn_new_savetest")#+str(int(time.time())))

def save_sess(sess,path):
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	save_path = saver.save(sess, path)
	print("Model saved in path: %s" % save_path)
	
def restore_sess(sess,path):
	saver = tf.train.Saver()
	saver.restore(sess, path)

def loadModel(vars,path):
	with tf.Session() as sess:
		learning_rate,epochs,batch_size,x,y,y_,y_clipped,cross_entropy,optimiser = vars
		
		init_op = tf.global_variables_initializer()

		# define an accuracy assessment operation
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
		# add a summary to store the accuracy
		tf.summary.scalar('accuracy', accuracy)
	
		merged = tf.summary.merge_all()
		print("starting")
	
		restore_sess(sess,path)
	
		data_test,labels_test = dataset.getTestData()
		print("Accuracy:",sess.run(accuracy, feed_dict={x: data_test, y: labels_test}))
		
if __name__ == "__main__":
	# run_simple_graph()
	# run_simple_graph_multiple()
	# simple_with_tensor_board()

# 	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	path_train  = "/home/henry/workspace/SSE3/neural_chesspiece/data_validation/"
	dataset,i_size,o_size = load_dataset(path_train)
	vars=create_vars(i_size,(400,),o_size, epochs = 1000, batch_size = 20)
# 	loadModel(vars,"/home/henry/workspace/SSE3/neural_chesspiece/nn_new_savetest")
	train_network(dataset,vars,lrate = .005, learning_rate_slow_param = 500.0)
	#Epoch: 661  cost = 0.16895  -.0002 Accuracy: 0.97637796
	#Epoch: 151  cost = 0.65307  Accuracy: 0.9133858
		