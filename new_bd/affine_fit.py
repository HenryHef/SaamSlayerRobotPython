from numpy.random.mtrand import randint
import numpy as np

def fit_affine(X,Y):
	Y=np.array([[y[0],y[1],1.0] for y in Y]).transpose()
	X=np.array([[x[0],x[1],1.0] for x in X]).transpose()
	print(X)
	print(Y)
	print(Y.dot(np.linalg.inv(X)))
	return Y.dot(np.linalg.inv(X))

def fit_perspective_transform(X,A,guess):
	X_=np.zeros((len(X),3))
	for i in range(len(X)):
		X_[i][0]=X[i][0]
		X_[i][1]=X[i][1]
		X_[i][2]=1.0
	X=np.array(X_)
	A=np.array(A)
	def y(M):
		re=0
		for n in range(X.shape[0]):
			u=M.dot(X[n])
			u=[u[0]/u[2],u[1]/u[2]]
			re+=(u[0]-A[n][0])**2+(u[1]-A[n][1])**2
		return re
	def p_y_p_Mij(M,i,j):
		delta=.001
		delMij=np.zeros(M.shape)
		delMij[i,j]=delta
		return (y(M+delMij)-y(M))/delta
	
	def grad_y_by_M(M):
		re=np.zeros(M.shape)
		for i in range(M.shape[0]):
			for j in range(M.shape[1]):
				re[i,j]=p_y_p_Mij(M,i,j)
		return re
	
	M=guess
# 	print(M,y(M))
# 	M=np.array([[2,0,0],[0,2,0],[0,0,1]])
# 	print(M,y(M))
# 	M=np.array([[.01,0,0],[0,.01,0],[.01,0,.01]])
# 	print(M,y(M))
	ly=y(M)
	for i in range(0):
		a,b=randint(0,3),randint(0,3)
		print (a,b)
		grad = p_y_p_Mij(M, a,b)
		gradM=np.zeros((3,3))
		gradM[a,b]=1
		print (grad,gradM)
		M_=M-.1*y(M)/grad*gradM
		print(y(M))
		print(M)
# 		if(i>100 and y(M_)<y(M)):
# 			step/=2.0
# 			print (step)
		M=M_
# 			step/=2.0
# 			print("divstep,",step)
# 		else:
# 			M=M_
# 			ly=ny
# 			step=.5
# 		print(y(M))
# 		M/=M[2,2]
	return M
from_pt = [[184.0, 348.0], [234.0, 345.0], [283.0, 342.0], [333.6666666666667, 340.0], [381.0, 336.0], [431.0, 333.0], [185.0, 299.0], [232.0, 297.0], [280.0, 294.0], [329.0, 292.0], [185.0, 254.0], [231.0, 252.0], [277.0, 249.0], [325.0, 247.0], [371.0, 244.0], [417.0, 242.0], [183.66666666666666, 212.33333333333334], [230.0, 209.0], [277.0, 207.0], [321.0, 204.0], [366.0, 202.0], [185.0, 172.0], [228.0, 170.0], [273.0, 168.0], [317.0, 165.0], [361.0, 163.0], [405.0, 161.0]]
to_pt = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
M=fit_perspective_transform(from_pt,to_pt,
	fit_affine([[184.0, 348.0], [234.0, 345.0], [185.0, 299.0]],[[0, 0], [0, 1], [1, 1]]))
# M=fit_affine([[184.0, 348.0], [234.0, 345.0], [185.0, 299.0]],[[0, 0], [0, 1], [1, 1]])
# from_pt=[[0,0],[1,0],[1,1],[0,1],[3,2]]
# to_pt=[[0,0],[2,0],[2,2],[0,2],[6,4]]
# M=fit_perspective_transform(from_pt,to_pt)
def proj(M,v):
	v=M.dot(np.array([v[0],v[1],1.0]))
	return (v[0]/v[2],v[1]/v[2])
def inv_proj(M,v):
	v=np.linalg.inv(M).dot(np.array([v[0],v[1],1.0]))
	return (v[0]/v[2],v[1]/v[2])
print(proj(M,from_pt[0]))
print(proj(M,from_pt[1]))
print(proj(M,from_pt[2]))
print(proj(M,from_pt[6]))


						
