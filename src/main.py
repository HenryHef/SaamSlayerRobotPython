import time
# from detector import detect_img
# import chess_engin as ce
import cv2 as cv
from robot import Robot, startAndGetCMDThread
# from camera import Camera
# from photostream_manager import create_training_data
import numpy as np



# img = cv.imread("/home/henry/workspace/SSE3/neural_chesspiece/images/img_1521220368.jpg")
# cam = Camera()
# croped_img,board = detect_img(img)

#import detector as dec
def readAndPrintImg():
	r = Robot()
	cmder=startAndGetCMDThread(r)
	cmder.sendPickUpPieceSq((1,1))
	cmder.sendPutPieceSq((2,4))
	
	stream1 = cv.VideoCapture (1); #0 is the id of video device.0 if you have only one camera.
	
	if not stream1.isOpened():
		print("cannot open camera")
	print("/home/henry/workspace/SSE3/neural_chesspiece/images/img_"+str(int(time.time())))
	while (True):
		cameraFrame = stream1.read()[1];
		cv.imshow("cam", cameraFrame);
		if (cv.waitKey(30) >= 0):
			break;
	tm = str(int(time.time()))
	cv.imwrite("/home/henry/workspace/SSE3/neural_chesspiece/images/img_"+tm+".jpg", cameraFrame);
def readVideo():
	cap = cv.VideoCapture('vtest.avi')
	
	while(cap.isOpened()):
		ret, frame = cap.read()
	
		gray = cv.cvtColor(frame, cv.cv2.COLOR_BGR2GRAY)
	
		cv.imshow('frame',gray)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv.destroyAllWindows()
def recordVideo():
	cap = cv.VideoCapture(1)
	
	# Define the codec and create VideoWriter object
	fourcc = cv.VideoWriter_fourcc(*'XVID')
	out = cv.VideoWriter('output2.avi',fourcc, 20.0, (640,480))
	
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret==True:
	
			# write the flipped frame
			out.write(frame)
	
			cv.imshow('frame',frame)
			if cv.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break
	
	# Release everything if job is finished
	cap.release()
	out.release()
	cv.destroyAllWindows()
recordVideo()
