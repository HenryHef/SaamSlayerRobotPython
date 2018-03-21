import numpy as np
from utils import image_transform

chess_square_size = 2.25*2.54
chess_board_size = 8*chess_square_size

chess_board_points = chess_board_size*np.float32([
	[0,0,0],
	[8,0,0],
	[8,8,0],
	[0,8,0]])

chess_board_padded_points = chess_board_size*np.float32([
	[-.25	,-.25	,0],
	[8.5,-.25,0],
	[8.5,8.25,0],
	[-.25,8.25,0]])

class Board:
	def __init__(self,transmat,square_pix_size):
		self.transmat=transmat
		self.invmat=np.linalg.inv(transmat)
		self.square_pix_size=square_pix_size
	def transform_boardpix_imagepix(self,pt):
		ptn = self.invmat.dot(np.array([pt[0],pt[1],1.0]))
		ptn = [ptn[0]/ptn[2],ptn[1]/ptn[2]]
		return ptn
	def transform_squarenum_imagepix(self,pt):
		ptn = self.invmat.dot(np.array([self.square_pix_size*pt[0],self.square_pix_size*pt[1],1.0]))
		ptn = [ptn[0]/ptn[2],ptn[1]/ptn[2]]
		return ptn
	
class PaddedCroppedBoard:
	def __init__(self,board,uncroped_img,square_pix_size,padding):#minx,maxx,miny,maxy
		unshifted_board_coors = np.float32([
			[-padding[0], -padding[2]],
			[8+padding[1], -padding[2]],
			[8+padding[1], 8+padding[3]],
			[-padding[0], 8+padding[3]]
			])
		shifted_board_coors = np.float32([
			[0, 0],
			[8+padding[0]+padding[1], 0],
			[8+padding[0]+padding[1], 8+padding[2]+padding[3]],
			[0, 8+padding[2]+padding[3]]
			])
		
		image_points = [board.transform_squarenum_imagepix(pt) for pt in unshifted_board_coors]
		
		self.croped_img, M = image_transform(uncroped_img,image_points,
											square_pix_size,shifted_board_coors)
		self.padding=padding
		self.square_pix_size=square_pix_size
		
		
		
	def sliceXXYY(self,ptl,size):#i.e (.5,.5),(2.5,3.5)
		print (ptl,size)
		minX,maxX,minY,maxY=ptl[0] , ptl[0]+size[0] , ptl[1] ,ptl[1]+size[1]
		minX,maxX,minY,maxY=minY,maxY,minX,maxX
		minX,maxX=8-maxX,8-minX
		print (minX,maxX,minY,maxY)
		#minY,maxY=8-maxY,8-minY
		minX+=self.padding[2]
		maxX+=self.padding[2]
		minY+=self.padding[0]
		maxY+=self.padding[0]
		
		minX*=self.square_pix_size
		maxX*=self.square_pix_size
		minY*=self.square_pix_size
		maxY*=self.square_pix_size
		print (minX,maxX,minY,maxY)
		print(self.croped_img.shape)
		return self.croped_img[int(minX):int(maxX),int(minY):int(maxY)]
	
	
	
	