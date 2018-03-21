import cv2 as cv
from utils import image_transform
from chessboard import PaddedCroppedBoard
from chess_engin import KING,QUEEN,ROOK,KNIGHT,BISHOP,PAWN,NO_PIECE,WHITE,BLACK,NO_PLAYER
import numpy as np
from detector import detect_img
import os

def process_frame(img,board):
	#crop image
	#first use perspectave to crop to a -.25,-.25    8.25,8.5  image
	square_length_pix=64
	board.square_pix_size=square_length_pix
	
	pMinX,pMaxX,pMinY,pMaxY=.5,.5,1.0,.5
	pcboard  = PaddedCroppedBoard(board,img,64,(pMinX,pMaxX,pMinY,pMaxY))
	
	cv.imshow("croped img",pcboard.croped_img)
	cv.imshow("uncroped img",img)
	cv.imshow("a2",pcboard.sliceXXYY((0,1),(1,2)))
	for row in range(8):
		for col in range(8):
			pcboard.sliceXXYY((row-.25,col-.25),(1.5,1.5))
	while True:
		if(cv.waitKey(30)>=0):
			break

def create_training_data(img_path,label,direc):
	
	img = cv.imread(img_path)
	croped_img,board = detect_img(img)
	name=img_path[-18:-4]
	#crop image
	#first use perspectave to crop to a -.25,-.25    8.25,8.5  image
	square_length_pix=64
	board.square_pix_size=square_length_pix
	
	pMinX,pMaxX,pMinY,pMaxY=.5,.5,1.0,.5
	pcboard  = PaddedCroppedBoard(board,img,64,(pMinX,pMaxX,pMinY,pMaxY))
	
# 	lmap={"K":(KING,WHITE),
# 		"Q":(QUEEN,WHITE),
# 		"R":(ROOK,WHITE),
# 		"B":(BISHOP,WHITE),
# 		"N":(KNIGHT,WHITE),
# 		"P":(PAWN,WHITE),
# 		"k":(KING,BLACK),
# 		"q":(QUEEN,BLACK),
# 		"r":(ROOK,BLACK),
# 		"b":(BISHOP,BLACK),
# 		"n":(KNIGHT,BLACK),
# 		"p":(PAWN,BLACK),
# 		".":(NO_PIECE,NO_PLAYER),
# 		"X":'X'}
	for k in ["K","Q","R","B","N","P","_","k","q","r","b","n","p"]:
		path=direc+k+"/"
		if not os.path.exists(path):
			os.makedirs(path)
	for row in range(8):
		for col in range(8):
			lb = label[col*9+row]
			if(lb=='X'):
				continue
			timg = pcboard.sliceXXYY((row-.25,col-.25),(1.5,2))
			#write file to directory
			path=direc+(lb if lb!="." else "_")+"/"
			cv.imwrite(path+name+"_"+str(row)+str(col)+".png",timg)


def start_stream():
	stream1 = cv.VideoCapture (1); #0 is the id of video device.0 if you have only one camera.
	
	if not stream1.isOpened():
		print("cannot open camera")
	print("/home/henry/workspace/SSE3/neural_chesspiece/images/img_"+str(int(time.time())))
	while (True):
		cameraFrame = stream1.read()[1];
		process_frame(cameraFrame)
		
		
		
		