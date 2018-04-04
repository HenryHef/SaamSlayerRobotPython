# Copyright (c) 2018, Henry Heffan
# All rights reserved.

import cv2 as cv
from detector import detect_img
from chessboard import PaddedCroppedBoard
import os

def create_training_data(img_path,label,direc):
	print("IMG PTH",img_path)
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
			

img_path = "/home/henry/workspace/SSE3/neural_chesspiece/images/"
opath = "/home/henry/workspace/SSE3/neural_chesspiece/data/"

dmap = {"img_1521220640.jpg":"R.brPpnR;.r.PrPn.;Q.Q.rN..;.N.B.QqN;b.K.nB.k;.b.p.kB.;.pR.Q.N.;.n.R.p.P",
		"img_1521220368.jpg":"..K.N..b;pN.kBr.b;.rpr.RK.;R..nb.k.;QNn..R.q;.pPQNnn.;PpR.Br.B;PP....Q.",
		"img_1521220418.jpg":"N.Kr..b.;pn.k.Nb.;r.prBnRK;...n....;RN.b.RAN;.pP.Pn.q;...PBr.B;.QpPkRQ.",
		"img_1521220519.jpg":".R..pNb.;n.nPP.BQ;.Q..NN.K;.bnR.qr.;Q..brp.R;rkK.R.P.;...k.PpN;rB..p.nB",
		"img_1521220800.jpg":"...bNr..;.N....b.;rk.PKQ.R;.pRQ.Pq.;nNrkRb.Q;pKPB.Rr.;Bn..p.nP;p.N.nB..",
		"img_1521221594.jpg":".K.RbNrN;n.k.R.Q.;.r.P.r..;n.n.Q.qB;.p.R.b..;..P.b.Np;Bp.Q.PRk;BnN.K.Pp",
		"img_1521221740.jpg":"...n..rN;n.N.RP..;.r.PBr.K;nbn.Q.k.;.p.Bkb.P;R.PqbRNp;.p.K..R.;QpN.BQ..",
		"img_1521551337.jpg":".Q..n.n.;.pRNpB.B;.r.k..pk;.QNbQNRB;.nb.R.R.;r.p.P..N;.P.P..b.;nPK.Qq.p",
		"img_1521551390.jpg":"k.N.Kn.B;...R.pp.;N.b.RR.n;.Q..kNNR;..b..bB.;..rQ.PPB;.r.rPqnp;.Q.n.PK.",
		"img_1521551552.jpg":"k..pr.R.;..K.r.N.;.pPrPp.n;.K..Q.pQ;.QBb.q.b;R..Pk.BP;RNn.B.rN;..NR.n.n",
		"img_1521551625.jpg":"kr.n.Np.;..pBpR..;QP.NK.pb;.B.nnQq.;.Pn..rNr;K..RbNk.;..bRB.PP;.Q..R...",
		"img_1521551755.jpg":"....QBn.;.KKP..pR;Nb..rR.N;..QB.nnN;qP.p.rb.;..pPrkRk;.NB.pb.n;...PQR..",
		#"img_1521551867.jpg":"",
		#"img_1521551989.jpg":"",
		"img_1521553414.jpg":"B..Q.r.K;.Nr.Pn..;.np.bRkP;..nR.PNR;nN...Qr.;..ppQ.PB;BbqBkN..;..Q.K.Rp",
		"img_1521553509.jpg":"Nn.rN.Bk;b.P..K..;.b.QX.qP;N.RPRX..;.P.p..Qp;nBp.KB.Q;bQ.RN.k.;.n.p..Rn",
		"img_1521553924.jpg":"....O.QB;.OpRRPp.;O..rO..P;.OO.rKR.;Q..N.BOQ;nN.nPP.p;R.K.b.Nn;.B.nN.Q.",
		"img_1521553967.jpg":".rQr.Bp.;kR.qp.Kb;pkRb.P..;Kn..r.BQ;.RbN.PN.;.N..n.Qp;nnNQ.P.P;..R...B.",
		"img_1521554078.jpg":".Rr.r..B;qp.Qpp..;.NQb.bnK;.nb.rP..;Rn.RKqpP;..N.kQ..;.Q.B.R.P;.NN.n.B.",
		"img_1521554168.jpg":"B.pRr..B;.Nk..pn.;nQ.rNbpK;.br...R.;Q.PRKq.n;.B.NRQ.P;Q.Pb..p.;.k.N.n.P",
		"img_1521554295.jpg":".RP.rQBn;kQ.RrN.P;..qK.pP.;NB.NP.nb;.b.b.Rn.;KQ..p.Q.;RkBp.N..;..n.p..r",
		"img_1521554403.jpg":"R.Pb..k.;Q.p.rR..;.nN.KPnr;.p.b..Rr;Nnp.QPQ.;..B..bRP;N.pNBn.Q;..B.q.Kk",
		"img_1521559385.jpg":"...P.B.P;.Q......;...N.Q.R;.N......;...K.Q.R;.B......;...K.Q.R;........",
		"img_1521559434.jpg":"........;.k...b..;...b...r;.p...q..;...n...r;.p...k..;...n...r;.n...p..",
	  "image_1522201333.jpg":"RNBQKBNR;PPPP.PPP;........;....P...;....p...;........;pppp.ppp;rnbqkbnr",
	 "img_run_000000000.jpg":"RNBQK..R;PPP..PPP;........;...P....;...qNP..;p.p..n..;.pp...pp;r.b.kb.r",
	  "img_run_00000001.jpg":"RNBQK..R;PPPP.PPP;.....N..;....P...;.B..p...;........;pppp.ppp;rnbqkbnr"
}
		
		



# dmap={"img_1521220640.jpg":"R.brPpnR;.r.PrPn.;Q.Q.rN..;.N.B.QqN;b.K.nB.k;.b.p.kB.;.pR.Q.N.;.n.R.p.P"}
for im in dmap:
	create_training_data(img_path+im
					,dmap[im]
					,opath)		

img_path = "/home/henry/workspace/SSE3/neural_chesspiece/images/validation/"
opath = "/home/henry/workspace/SSE3/neural_chesspiece/data_validation/"

dmap = {"img_1521222205.jpg":".p.B.R.P;Pr..BbN.;p.QN.K.r;.nRbPBrR;Q..k..n.;.R.nnkpK;.qp..QN.;...N.b.P",
		"img_1521222685.jpg":"Kp.P.n.K;..B.Qk.n;.Nk.R.Qp;Q..R.N..;PbqrB.bR;P.B.pP.n;...N.Nn.;rpR.b.r."
		}

for im in dmap:
	create_training_data(img_path+im
					,dmap[im]
					,opath)	
	
	


