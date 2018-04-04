from chess_engin import KING, QUEEN, ROOK, KNIGHT, BISHOP, PAWN, NO_PIECE, WHITE, BLACK, NO_PLAYER
from chessboard import PaddedCroppedBoard
import cv2 as cv
from data_loader import datavec_from_image
from detector import detect_img
from keras_nn import getModelPredFunc
import numpy as np


# from utils import image_transform
# import os
CASTLING_P_KSQ_4SQ_MAP={(WHITE,True):set([(0,4),(0,5),(0,6),(0,7)]),
					(WHITE,False):set([(0,4),(0,3),(0,2),(0,0)]),
					(BLACK,True):set([(7,4),(7,5),(7,6),(7,7)]),
					(BLACK,False):set([(7,4),(7,3),(7,2),(7,0)])}
CASTLING_P_KSQ_MOVE_MAP={(WHITE,True):(0,4,0,6),
					(WHITE,False):(0,4,0,2),
					(BLACK,True):(7,4,7,6),
					(BLACK,False):(7,4,7,2)}

def calcPercent(img,val):
	percent=(img == val).sum()/1.0/(np.prod(img.shape))
	return percent
# 				print("--",i,j,"--")

def regress_move_from_cg_maps(cg,nn_map,delta_map):
	bstates=cg.getValidChildStates()
	to_cproba = lambda x:(2.0/-(x/2.0+2.0)+1)*(np.arctan(2.0*x-4.0)/np.pi+.5)
	actual_psq_map={}
# 	
	print("NN BAORD:")
	for i in range(8):
		i=7-i
		l=""
		for j in range(8):
			l+=str(nn_map[(i,j)][0])+" "
		print(l)
# 	
# 	print("CG BAORD:")
# 	for i in range(8):
# 		i=7-i
# 		l=""
# 		for j in range(8):
# 			l+=str(cg.get(i,j).getPlayer())+" "
# 		print(l)
	
	print("ACT BAORD:")
	for i in range(8):
		i=7-i
		l=""
		for j in range(8):
			actual_psq_map[(i,j)]= cg.get(i,j).getPlayer() if delta_map[(i,j)]<.5 else nn_map[(i,j)][0]
			l+=str(actual_psq_map[(i,j)])+" "
		print(l)
	
	print("")
	move=None
	for x in range(bstates.size()):
		state=bstates.get(x)
		eq=all(state.getPlayer(i,j)==actual_psq_map[(i,j)] for i in range(8) for j in range(8))
		mv=state.getMoveSSEE()
		if(eq):
			move=(mv[0],mv[1],mv[2],mv[3])
			return move
	return None

class FrameStreamProccesor:
	ct=0
	pcboard=None
	FRAME_RATE=30
	CROP_FRAME_DELAY=1*FRAME_RATE
	STABLE_PROCESS_FRAME_DELAY=1.25*FRAME_RATE
	prev_croped_frame=None
	first_croped_frame = None
	stable_ct=0
	destabilized_bool=False
	cframe=None
	
	def __init__(self,chess_game):
		self.cg=chess_game
		self.proc_ct=0
	
	def attempt_process_stable_frame(self,frame,diff,threshed,od):
		print("ATTEMPTING FRAME PROC")
		cv.imwrite("img_run_00000001.jpg",frame)
		self.proc_ct+=1
		dpcboard = self.pcboard.copy()
		dpcboard.set_img_precroped(diff)
# 		cv.imshow("diffed",diff)
		l=[]#pt,percent
		
		for i in range(8):
			for j in range(8):
				sl=dpcboard.sliceXXYY((i-.15,j-.15), (1+.3,1+.5))
# 				cv.imshow("sl",sl)
				sl=sl.astype(np.float32)
				sl=np.clip(sl,0,30)
# 				cv.waitKey(0)
				sl=np.power(sl,3).sum()/255.0/(np.prod(sl.shape))
				sl=np.sum(sl)
				l.append(((j,i),sl))
		lsum=sum(e[1] for e in l)
		change_proba_list=l
		change_proba_map={e[0]:e[1] for e in l}
		change_proba_list.sort(key=lambda x:x[1],reverse=True)
		
		print("change_proba_list  ",change_proba_list)
		
		SQUARE_CHANGED_THRESHOLD = .1
		changed_set=set(1 if e[1]>SQUARE_CHANGED_THRESHOLD else 0 for e in l)
		
		print(l,len(changed_set),changed_set)

		predor = getModelPredFunc()
		fpcboard = self.pcboard.copy()
		fpcboard.set_img(frame)
		sq_p_proba_map={}
		vcmap={WHITE:"W",BLACK:"B",NO_PLAYER:"."}
		for i in range(8):
# 			line=""
			for j in range(8):
				sl=fpcboard.sliceXXYY((j-.25,(7-i)-.25), (1.5,2))
				vec=datavec_from_image(sl)
				pred,proba,probam=predor(vec)
				sq_p_proba_map[(7-i,j)]=(pred,proba)
# 				line+=vcmap[pred]+" "
# 			print( line)
		move = regress_move_from_cg_maps(self.cg,sq_p_proba_map,change_proba_map)
		print(move)
		self.cg.applyMoveAlgebraic(move[0],move[1],move[2],move[3])
		self.cg.printBoard()
# 		#decide if the move is castling.
# 		if(len(changed_set)>=4):
# 			#4 squares were changed, so check that they were 4 castleable squares
# 			for key in CASTLING_P_KSQ_4SQ_MAP:
# 				player,kingsideQ=key
# 				rS,cS,rE,cE=CASTLING_P_KSQ_MOVE_MAP[key]
# 				if(CASTLING_P_KSQ_4SQ_MAP[key].issubset(changed_set) and \
# 						self.cg.shouldConsiderCastlePlayerKingsideQ(player,kingsideQ) and \
# 						self.cg.isValidMoveAlgebraic(rS,cS,rE,cE)):
# 					self.cg.applyMoveAlgebraic(rS,cS,rE,cE)
# 					print("CASTLING")
# 					self.cg.printBoard()
# # 		ls=[e[1] for e in l]
# 		l=l[:2]
# # 		print(ls)
# 		cS,rS=l[0][0][0],l[0][0][1]
# 		cE,rE=l[1][0][0],l[1][0][1]			
# # 		print(l[0][0][0]+1,l[0][0][1]+1," and ",l[1][0][0]+1,l[1][0][1]+1)
# 		forwValid = self.cg.isValidMoveAlgebraic(rS,cS,rE,cE)
# 		backValid = self.cg.isValidMoveAlgebraic(rE,cE,rS,cS)
# 		if(forwValid and not backValid):
# 			self.cg.applyMoveAlgebraic(rS,cS,rE,cE)
# 		elif(backValid and not forwValid):
# 			self.cg.applyMoveAlgebraic(rE,cE,rS,cS)
# 		else:
# 			print("BAD MOVE")
# 		self.cg.printBoard()

		
		
		self.destabilized_bool=False
		self.first_croped_frame=self.pcboard.croped_img
	
	def process_frame(self,frame):
		self.cframe=frame
		self.ct+=1
		if(self.ct==self.CROP_FRAME_DELAY):
			croped_img,board = detect_img(frame)
			pMinX,pMaxX,pMinY,pMaxY=.5,.5,1.0,.5
			self.pcboard  = PaddedCroppedBoard(board,frame,64,(pMinX,pMaxX,pMinY,pMaxY))
			self.prev_croped_frame=self.pcboard.croped_img
			self.first_croped_frame=self.pcboard.croped_img
			print("\n"*10)
			return
		elif(self.ct<self.CROP_FRAME_DELAY):
			return
		
			
		self.pcboard.set_img(frame)
	# 	print(frame.dtype)
		
		cv.imshow("croped img",self.pcboard.croped_img)
		
	# 	diff=((prev_croped_frame.astype(np.float32)-pcboard.croped_img.astype(np.float32))/2+128.0).astype(np.uint8)
		
		diff=(np.sum(np.abs(((self.first_croped_frame.astype(np.float32)
							-self.pcboard.croped_img.astype(np.float32)))),axis=2)/3.0).astype(np.uint8)
		
		diff_b= cv.blur(diff,(9,9))
	# 	tblured= cv.blur(,(5,5))
		cv.imshow("diff",diff_b)
		ret,th1 = cv.threshold(diff_b,25,255,cv.THRESH_BINARY)
		od=th1.copy()
		percent=calcPercent(th1,255)
		#TODO, require no movement for a step to be called stable
		if(percent<.035):
			self.stable_ct+=1
		else:
			self.stable_ct=0
			self.destabilized_bool=True
# 		print(percent,self.stable_ct)
		if(self.stable_ct>self.STABLE_PROCESS_FRAME_DELAY and self.destabilized_bool==True):
			self.attempt_process_stable_frame(frame,diff_b,th1,od)
		cv.circle(od,(10,10),10,(self.stable_ct*10,),5)

if __name__ == "__main__":
	import main
		
		