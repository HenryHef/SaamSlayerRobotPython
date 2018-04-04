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
	
class PaddedCroppedBoard:
	def __init__(self,board,uncroped_img,square_pix_size,padding):#minx,maxx,miny,maxy
		self.board=board
		unshifted_board_coors = np.float32([
			[-padding[0], -padding[2]],
			[8+padding[1], -padding[2]],
			[8+padding[1], 8+padding[3]],
			[-padding[0], 8+padding[3]]
			])
		self.shifted_board_coors = np.float32([
			[0, 0],
			[8+padding[0]+padding[1], 0],
			[8+padding[0]+padding[1], 8+padding[2]+padding[3]],
			[0, 8+padding[2]+padding[3]]
			])
		
		self.image_points = [board.transform_squarenum_imagepix(pt) for pt in unshifted_board_coors]
		
		self.padding=padding
		self.square_pix_size=square_pix_size
		
		self.set_img(uncroped_img)
		
	def set_img(self,nimg):
		if(type(nimg)!=np.ndarray):
			return
		self.croped_img, M = image_transform(nimg,self.image_points,
											self.square_pix_size,self.shifted_board_coors)
	def set_img_precroped(self,nimg):
		if(type(nimg)!=np.ndarray):
			return
		self.croped_img=nimg
		
	def sliceXXYY(self,ptl,size):
		minX,maxX,minY,maxY=self.sliceXXYYZone(ptl,size)
		return self.croped_img[int(minX):int(maxX),int(minY):int(maxY)]
	def sliceXXYYZone(self,ptl,size):#i.e (.5,.5),(2.5,3.5)
# 		print (ptl,size)
		minX,maxX,minY,maxY=ptl[0] , ptl[0]+size[0] , ptl[1] ,ptl[1]+size[1]
		minX,maxX,minY,maxY=minY,maxY,minX,maxX
		minX,maxX=8-maxX,8-minX
# 		print (minX,maxX,minY,maxY)
		#minY,maxY=8-maxY,8-minY
		minX+=self.padding[2]
		maxX+=self.padding[2]
		minY+=self.padding[0]
		maxY+=self.padding[0]
		
		minX*=self.square_pix_size
		maxX*=self.square_pix_size
		minY*=self.square_pix_size
		maxY*=self.square_pix_size
		return (minX,maxX,minY,maxY)
	def copy(self):
		re=PaddedCroppedBoard(self.board,None,self.square_pix_size,self.padding)
		re.set_img_precroped(self.croped_img)
		return re
	
	
	
	