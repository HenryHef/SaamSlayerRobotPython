# from chess_engin import KING,QUEEN,ROOK,KNIGHT,BISHOP,PAWN,NO_PIECE,WHITE,BLACK,NO_PLAYER
import cv2
import numpy as np

KING = 0
QUEEN = 1
ROOK = 2
BISHOP = 3
KNIGHT = 4
PAWN = 5
NO_PIECE = 6
WHITE = 0
BLACK = 1
NO_PLAYER = 2

# lab_dir_map = {"K":(KING,WHITE),
# 			"Q":(QUEEN,WHITE),
# 			"R":(ROOK,WHITE),
# 			"B":(BISHOP,WHITE),
# 			"N":(KNIGHT,WHITE),
# 			"P":(PAWN,WHITE),
# 			"_":(NO_PIECE,NO_PLAYER),
# 			"k":(KING,BLACK),
# 			"q":(QUEEN,BLACK),
# 			"r":(ROOK,BLACK),
# 			"b":(BISHOP,BLACK),
# 			"n":(KNIGHT,BLACK),
# 			"p":(PAWN,BLACK)}
lab_dir_map = {"K":1,
			"Q":1,
			"R":1,
			"B":1,
			"N":1,
			"P":1,
			"_":0,
			"k":2,
			"q":2,
			"r":2,
			"b":2,
			"n":2,
			"p":2}

def datavec_from_image(im):
# 	print("SHAPE:",im.shape)
	if(im.shape[-1]==3):
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# 	print("SHAPE2:",im.shape)
	hog = cv2.HOGDescriptor()
# 	print("DID HOG DES")
	h = hog.compute(im)
# 	h=np.array([h[i][0] for i in range(h.shape[0])])
	h=np.array(h)[:,0]
# 	print("MADE HOG V")
	
	vec=im.flatten()/255.0
	t_input = np.concatenate([vec, h])
	return t_input

import os
def load_image_arrays(direc,cap=400,onehot=True):
	if(cap==-1):
		cap=100000000
	input_list = []
	filename_list=[]
	lab_list = []
	
	subdirs = os.walk(direc)
	i_size=-1
	class_ct=[0,0,0]
	
	for subdir,subsubdirs,subdir_files in subdirs:
		ict = 0
# 		if not (subdir[-1]=="K" or subdir[-1]=="_"):
# 			continue
		for file in subdir_files:#process and add file
			ict+=1
			if(ict>cap):
				break
			path=subdir+"/"+file
			print(path)
			filename_list.append(path)
			im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
			t_input = datavec_from_image(im)
# 			t_input=h
			
			i_size = t_input.shape[0]
			
			input_list.append(t_input)
			lab_list.append(lab_dir_map[subdir[-1]])
			class_ct[lab_dir_map[subdir[-1]]]+=1
			
	lab_ct = max(lab_list)+1
	one_hot_lab_list = []
	for lab in lab_list:
		nlab = [0]*lab_ct
		nlab[lab]=1
		one_hot_lab_list.append(nlab)
		
	input_list = np.array(input_list)
	filename_list=np.array(filename_list)
	output_list = np.array(one_hot_lab_list) if onehot else np.array(lab_list)
	perm = np.random.permutation(input_list.shape[0])
	input_list=input_list[perm]
	output_list=output_list[perm]
	filename_list=filename_list[perm]
	print("CLASS CT:",class_ct)
	return input_list,output_list,filename_list,i_size,lab_ct

class DataSet:
	def __init__(self,features,labels,filenames,percent=.2, overlap=False):
		
		if len(labels.shape)>1:
			class_ct=np.zeros(labels.shape[1])
			for label in labels:
				class_ct+=label
			print("CLASS CT 2",class_ct,labels==1)
		#if overlap, then slice train = all, slice test = percent
		features=np.array(features)
		labels=np.array(labels)
		split = int(len(features)*(1-percent))
		
		if(overlap):
			self._data_train = features.copy()
			self._labels_train = labels.copy()
			
			self._data_test = features.copy()[split:]
			self._labels_test = labels.copy()[split:]
		else:
			self._data_train = features.copy()[:split]
			self._labels_train = labels.copy()[:split]
			
			self._data_test = features.copy()[split:]
			self._labels_test = labels.copy()[split:]
			
		self._data_all=features.copy()[:]
		self._labels_all=labels.copy()[:]
		self._filenames=filenames.copy()[:]
		self.train_batch_index = 0
	def getFNs(self):
		return self._filenames
	def newEpoch(self):
		perm = np.random.permutation(self.trainCt())
		self._data_train=self._data_train[perm]
		self._labels_train=self._labels_train[perm]
		self.train_batch_index = 0
		
	def trainCt(self):
		return self._data_train.shape[0]
	
	def isNextBatch(self):
		return self.train_batch_index<self.trainCt()
	
	def getNextBatch(self,max_size):
		size = max(max_size,self.trainCt()-self.train_batch_index)
		reData=self._data_train[self.train_batch_index:self.train_batch_index+size]
		reLabels=self._labels_train[self.train_batch_index:self.train_batch_index+size]
		self.train_batch_index+=size
		return reData,reLabels,size

	def getTestData(self):
		return self._data_test,self._labels_test

def load_dataset(direcTrain,ratio=.8,cap=-1,one_hot=True):
	features,labels,filename_list,i_size,o_size = load_image_arrays(direcTrain,cap,one_hot)
	assert features.shape[0] == labels.shape[0]
	
	return DataSet(features,labels,filename_list),i_size,o_size

def load_dataset_dimes(direcTrain,ratio=.8,cap=1,one_hot=True):
	features,labels,filename_list,i_size,o_size = load_image_arrays(direcTrain,cap,one_hot)
	assert features.shape[0] == labels.shape[0]
	
	return i_size,o_size

import tensorflow as tf
def load_tfdataset(direcTrain,ratio=.8,cap=-1):
	features,labels,filename_list,i_size,o_size = load_image_arrays(direcTrain,cap,False)
	assert features.shape[0] == labels.shape[0]
	return (features,labels)
# Load the training data into two NumPy arrays, for example using `np.load()`.


# path_train  = "/home/henry/workspace/SSE3/neural_chesspiece/data/"
# dataset_train,i_size,o_size = load_dataset(path_train)
# print( dataset_train[0].shape)


