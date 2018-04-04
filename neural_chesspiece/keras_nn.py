import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
import numpy
import numpy as np

from data_loader import load_dataset,load_dataset_dimes
from chess_engin import WHITE,BLACK,NO_PLAYER

seed = 7
numpy.random.seed(seed)
path_train  = "/home/henry/workspace/SSE3/neural_chesspiece/data/"
path_validate  = "/home/henry/workspace/SSE3/neural_chesspiece/data_validation/"

def create_model():
	i_size,o_size = load_dataset_dimes(path_train)
	model = Sequential()
	model.add(Dense(150, input_dim=i_size, activation='relu'))#200
	model.add(Dense(20, activation='relu'))#20
	model.add(Dense(o_size, activation='softmax'))
	return model

def load_trained_model(weights_path):
	model = create_model()
	model.load_weights(weights_path)
	return model
def train(path=None):
	model=None
	if(path!=None):
		model=load_trained_model(path)
	else:
		model = create_model()
	model.compile(loss='categorical_crossentropy', optimizer=
			keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
			metrics=['accuracy'])
	
	estimator = KerasClassifier(build_fn=lambda:model, epochs=50, batch_size=15, verbose=2)
	
	# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	checkpoint = ModelCheckpoint('keras_model/model-4.h5', verbose=1, monitor='val_loss',
								save_best_only=True, mode='auto')
	
	dataset,i_size,o_size = load_dataset(path_train)
	re=estimator.fit(dataset._data_all, dataset._labels_all,
						validation_split=0.2, callbacks=[checkpoint], verbose=True)
	print(re)

import cv2 as cv

mapping={0:NO_PLAYER,1:WHITE,2:BLACK}
__model=None
def getModelPredFunc():
	global __model
	if(__model==None):
		__model = load_trained_model("/home/henry/workspace/SSE3/neural_chesspiece/keras_model/model-4.h5")
	print("LOADED MODEL")
	def pred(vec):
# 		print(vec.shape)
		if(vec.shape[0]!=1):
			vec=numpy.expand_dims(vec, 0)
# 			print("now,",vec.shape)
		pred=__model.predict_classes(vec)[0]
		proba=__model.predict_proba(vec)[0]
		proba_str="x:{:.5f}".format(proba[0])+" W:{:.5f}".format(proba[1])+" B:{:.5f}".format(proba[2])
		return mapping[pred],proba,max(proba)
	return pred

if __name__ == "__main__":
# 	train("/home/henry/workspace/SSE3/neural_chesspiece/keras_model/model-4.h5")
	dataset,i_size,o_size = load_dataset(path_train)
	
	vcmap={WHITE:"W",BLACK:"B",NO_PLAYER:"x"}
	predictor = getModelPredFunc()
	print(dataset._data_all.shape)
	print(dataset._data_all[0].shape)
	for i in range(dataset._data_all.shape[0]):
		pred,proba,mp=predictor(dataset._data_all[i])
		act=dataset._labels_all[i].dot(np.array([0,1,2]))
		act=mapping[act]
		if(mp<.9):
			print("PRED:",vcmap[pred],", ACT:",vcmap[act],", PROBS:",proba)
		if(False and pred!=act):
			print(dataset._filenames[i][-30:],", PRED:",vcmap[pred],", ACT:",vcmap[act],", PROBS:",proba)
			im=cv.imread(dataset._filenames[i])
			cv.imshow("broken",im)
			cv.waitKey(0)

		


		
