import math

import numpy as np
from ortho_regress import orthoregress

def intersection(p1, p2):
	"""solve intersection"""
	#print("p1:",p1,"  p2:",p2)
	x =  (p2[1] -p1[1])/(p1[0]- p2[0])
	y = p1[0] * x + p1[1]
	return (x,y)

def corssproduct(v1,v2):
	return (v1[0]*v2[1]) - (v1[1]*v2[0])
def dotproduct(v1, v2):
	##print ("v1:",v1,"  v2:",v2)
	return v1[0]*v2[0]+v1[1]*v2[1]

def length(v):
	return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
# 	#print("v1:",v1,"  v2:",v2)
	inner = dotproduct(v1, v2) / (length(v1) * length(v2))
	if(inner>.9999):
		return 0.0
	if(inner<-.9999):
		return np.pi
	return math.acos(inner)

def get_four_corners(centered_grid):
	#really should use a matrix regresion, but line edge regressions work for now
	re=[]
	top,bot,left,right = [],[],[],[]
	for i in range(6):
		if((i,0) in centered_grid):
			bot.append(centered_grid[(i,0)][0])
		if((i,6) in centered_grid):
			top.append(centered_grid[(i,6)][0])
		if((0,i) in centered_grid):
			left.append(centered_grid[(0,i)][0])
		if((6,i) in centered_grid):
			right.append(centered_grid[(6,i)][0])
	top=orthoregress([x[0] for x in top],[y[1] for y in top])
	bot=orthoregress([x[0] for x in bot],[y[1] for y in bot])
	left=orthoregress([x[0] for x in left],[y[1] for y in left])
	right=orthoregress([x[0] for x in right],[y[1] for y in right])
	if((0,0) in centered_grid):
		re.append(centered_grid[(0,0)][0])
	else:
		re.append(intersection(left,bot))
	
	if((6,0) in centered_grid):
		re.append(centered_grid[(6,0)][0])
	else:
		re.append(intersection(right,bot))
	
	if((6,6) in centered_grid):
		re.append(centered_grid[(6,6)][0])
	else:
		re.append(intersection(right,top))
	
	if((0,6) in centered_grid):
		re.append(centered_grid[(0,6)][0])
	else:
		re.append(intersection(left,top))
	
	#print("GRID POINTS ARE",re)
	
	return re
			
			


def getGridFromPoints(pts, cam, parr_tol = 20*np.pi/180.0, perp_tol = 20*np.pi/180.0, same_len_tol = .7):
	#print("PTS:",pts)
	#first build a table from point to 4 closest nieghbors
	point_opoint_dist_map={}#{pt:[(opt,distSq),...]}
	for pt in pts:
		point_opoint_dist_map[pt]=[]
	for pt in pts:
		for opt in pts:
			if(pt==opt):
				continue
			point_opoint_dist_map[pt].append((opt,(pt[0]-opt[0])**2+(pt[1]-opt[1])**2))
	
	for pt in pts:
		point_opoint_dist_map[pt].sort(key=lambda x:x[1])
		point_opoint_dist_map[pt]=point_opoint_dist_map[pt][:4]
	#print("GRID:",point_opoint_dist_map)
	
	#then check pairwise
	n_map = {}#will be of form pt:[(dispVec,opt)] len<=4
	for pt in pts:
		n_map[pt]=[]
	for pt in pts:
		for niegb,distSq in point_opoint_dist_map[pt]:
			for n_niegb,n_distSq in point_opoint_dist_map[niegb]:
				if(n_niegb==pt):
					#niegb is good
					n_map[pt].append(((niegb[0]-pt[0],niegb[1]-pt[1]),niegb))
	n_map_cleaned = {}
	for pt in n_map:
		if(len(n_map[pt])!=0):
			n_map_cleaned[pt]=n_map[pt]
	n_map=n_map_cleaned
	#print("\n\nN_GRID:")
	for pt in n_map:
		pass
		#print(pt," : ",n_map[pt])
	#print ("\n\n")
			
	#then find those that are 4, close to perp, close same dist
	#looks for a point that has 4 nieghbors, in which, pick 1 and there is 1 close
	#to parrelel of the same lenth, and a second set that is close to perperdicular
	#that are the same length
	start_pt = None
	for pt in n_map:
		if(len(n_map[pt])==4):
			vecs = n_map[pt]
			#then check semiperp
			mainVec = vecs[0][0]
			isGood = True
			perpLen=-1
			perpAng=-1
			for vec,opt in vecs[1:]:
				ang = angle(mainVec,vec)
				if(ang>np.pi-parr_tol):
					lenRat = length(vec)/length(mainVec)
					if(lenRat<same_len_tol or lenRat>1.0/same_len_tol):
						isGood=False
						break
				elif(abs(np.pi/2-ang)<perp_tol):
					if(perpLen==-1):
						perpLen=length(vec)
						perpAng=ang
					else:
						lenRat = length(vec)/perpLen
						if(lenRat<same_len_tol or lenRat>1.0/same_len_tol or np.abs(perpAng-ang)>parr_tol):
							isGood=False
							break
				else:
					isGood=False
					break
			if(isGood):
				start_pt=pt
	#print ("START_PT:",start_pt)
	
	grid = {}#(-7,-7)...(7,7):(pt,east_vec,north_vec)   note:east_vec = (1,0)
	#north_vec = (0,1)#i.e. coordiant axes
	edge_points = []
	closed_points = set()
	sev = n_map[start_pt][0][0]
	snv=None
	
	for v,_pt in n_map[start_pt]:
		#print ("sev:",sev,"  v:",v, "  abs(np.pi/2-angle(sev,v)):",abs(np.pi/2-angle(sev,v)),
			#"dotproduct(sev, v):",corssproduct(sev, v))
		if(abs(np.pi/2-angle(sev,v))<perp_tol and corssproduct(sev, v)>0):
			snv=v
			break
	if(snv==None):
		raise ValueError("ERROR bad coor system")
		return
	
	#print("\nNVEC:",snv,"  EVEC:",sev,"\n")
		
	grid[(0,0)]=(start_pt,sev,snv)
	
	pt_loc_map={}#point to location
	pt_loc_map[start_pt]=(0,0)
	edge_points.append(start_pt)
	
	
	def get_oriented(vec_pt_tup_list,e_vec,n_vec):
		n_vecs_tups=[]
		e_vecs_tups=[]
		for vec,pt in vec_pt_tup_list:
			if(angle(vec,n_vec)<parr_tol or angle(vec,n_vec)>np.pi-parr_tol):
				lenRat = abs(length(vec)/length(n_vec))
				if(lenRat<1.0/same_len_tol and lenRat>same_len_tol):
					#print("IS N VEC:",vec)
					n_vecs_tups.append((vec,pt))
			elif(angle(vec,e_vec)<parr_tol or angle(vec,e_vec)>np.pi-parr_tol):
				lenRat = abs(length(vec)/length(e_vec))
				if(lenRat<1.0/same_len_tol and lenRat>same_len_tol):
					#print("IS E VEC:",vec)
					e_vecs_tups.append((vec,pt))
				else:
					pass
					#print("REJECT EVEC ON LEN:",vec,lenRat,length(vec),length(e_vec),1.0/same_len_tol,same_len_tol,lenRat<1.0/same_len_tol,lenRat>same_len_tol)
			else:
				pass
				#print("REJECT VEC ON ANGLE:",vec,angle(vec,n_vec),angle(vec,e_vec),e_vec,n_vec,np.pi-parr_tol)
		re={}
		for nvec,pt in n_vecs_tups:
			if(dotproduct(nvec, n_vec)>0):
				re[(0,1)]=(nvec,pt)
			else:
				re[(0,-1)]=(nvec,pt)
			
		for evec,pt in e_vecs_tups:
			if(dotproduct(evec, e_vec)>0):
				re[(1,0)]=(evec,pt)
			else:
				re[(-1,0)]=(evec,pt)
		
		re_nvec,re_evec	= None,None
		if (0,1) in re:
			re_nvec=re[(0,1)][0]
		elif (0,-1) in re:
			re_nvec=re[(0,-1)][0]
			re_nvec=(-re_nvec[0],-re_nvec[1])
		else:
			re_nvec=n_vec
			
		if (1,0) in re:
			re_evec=re[(1,0)][0]
		elif (-1,0) in re:
			re_evec=re[(-1,0)][0]
			re_evec=(-re_evec[0],-re_evec[1])
		else:
			re_evec=e_vec
		
		return re,re_evec,re_nvec
			
		
	
	while(len(edge_points)>0):
		cedge_pt = edge_points[0]
		#print("EXPANDED:",cedge_pt," at loc ",pt_loc_map[cedge_pt])
		edge_points=edge_points[1:]
		if(cedge_pt in closed_points):
			continue
		e_vec = grid[pt_loc_map[cedge_pt]][1]
		n_vec = grid[pt_loc_map[cedge_pt]][2]
		#take the edge point, and for each other the other outgoing edges,
		#for each edge check that the lenths of the vectors are consistent with neerby vectors. 
		orientation_vec_pt_tups_map,nnvec,nevec = get_oriented(n_map[cedge_pt],e_vec,n_vec)
		#return orthaginal dir map to pair, missing if not there, plus new axes
		for grid_disp in orientation_vec_pt_tups_map:
			pix_disp,opt = orientation_vec_pt_tups_map[grid_disp]
			opt_loc = (grid_disp[0]+pt_loc_map[cedge_pt][0],grid_disp[1]+pt_loc_map[cedge_pt][1])
			if(opt in pt_loc_map):
				#Then check whether the connection is in the map.
				#IF it is, then check its location is consistent
				if not (pt_loc_map[opt] == opt_loc):
					#print("cpt:",cedge_pt,"  opt:",opt,"  opt local loop predicted loc:",opt_loc, "  global loc:",pt_loc_map[opt])
					#print("cpt loc:",pt_loc_map[cedge_pt])
					#print("grid:",grid)
					#print("pt_loc_map:",pt_loc_map)
					raise ValueError('INCONSISTENT MAP')
					return
			#otherwise add the point as an edge node.
			else:
				#print("ADDED:",opt," at loc ",opt_loc, "  from point ",cedge_pt," at loc:",pt_loc_map[cedge_pt]," with disp:",grid_disp)
				grid[opt_loc]=(opt,nnvec,nevec)
				pt_loc_map[opt]=opt_loc
				edge_points.append(opt)
		closed_points.add(cedge_pt)
		#take a point with the last north vector, and add it to the grid,
		#and to open points. validate
	#print ("FINAL GRID:",grid)
	
	#finally, take the grid, and find the dencest 7x7 region.
	den_list=[]
	for i in range(-6,1):#so from -6 to 0
		for j in range(-6,1):
			ct=0
			for x in range(i,i+7):
				for y in range(j,j+7):
					if((x,y) in grid):
						ct=ct+1
			den_list.append((i,j,ct))
	for i in range(-6,7):#so from -6 to 0
		l=""
		for j in range(-6,7):
			if((i,j) in grid):
				l=l+'x'
			else:
				l=l+'.'
		#print(l)
	
	#print(den_list)
	mx=max(den_list,key=lambda  x:x[2])
	#print(mx)
	
	centered_grid={(i,j):grid[(i+mx[0],j+mx[1])] for i in range(-6,7) for j in range(-6,7) if (i+mx[0],j+mx[1]) in grid}
	
	return get_four_corners(centered_grid)
	#TODO implement lines on edges, for missing corners, then, later, matrix regression

	