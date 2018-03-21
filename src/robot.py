import serial
import threading
import time

import numpy as np

from chessboard import chess_square_size


corner_offset=(4*2.54,8*2.54)
cen_sq_a1=(corner_offset[0]+chess_square_size/2,corner_offset[1]+chess_square_size/2)#(4*2.54,8*2.54)

def calc_angles_from_sq(robot, sq):
	r = (sq[0]*chess_square_size+cen_sq_a1[0],-(sq[1]*chess_square_size+cen_sq_a1[1]))
	R_sqd = (r[0]**2+r[1]**2)
	norm_r=np.sqrt(R_sqd)
	
	N=robot.armLen1
	M=robot.armLen2
	
	a2 = np.arccos((R_sqd-M**2-N**2)/(2*N*M))
	
	vna1 = (M*np.sin(a2),N+M*np.cos(a2))
	norm_vna1 = np.sqrt(vna1[0]**2+vna1[1]**2)
	
	a1 = np.arccos((r[0]*vna1[0]+r[1]*vna1[1])/(norm_vna1*norm_r))
	
	return(a1,a2)

class Pose:
	def __init__(self,a1,a2,hu,ho):
		self.angle1 = a1
		self.angle2 = a2
		self.hand_up = hu
		self.hand_open = ho

class Robot:
	def __init__(self):
		self.pose = Pose(0,0,False,False)
		self.armLen1 = 14.8*2.54#cm
		self.armLen2 = 14.25*2.54#cm  realy 14.625 to back of hand
		self.delay = [0,0,0,0]
		
	def sendState(self,ser):
		data = bytes([int(np.round(self.pose.angle1*180.0/np.pi)),
					int(np.round(self.pose.angle2*180.0/np.pi)),
					self.pose.hand_up,self.pose.hand_open])
		ser.write(data)
		
	def calc_sq_coor(self):
		offset = (self.armLen1*np.sin(self.pose.angle1) + self.armLen2*np.sin(self.pose.angle1+self.pose.angle2),
			-self.armLen1*np.cos(self.pose.angle1) - self.armLen2*np.cos(self.pose.angle1+self.pose.angle2))
		return ((offset[0]-cen_sq_a1[0])/chess_square_size,
			(offset[1]-cen_sq_a1[1])/chess_square_size)
			
	def set_angles_to_sq(self,sq):
		a1,a2 = calc_angles_from_sq(self,sq)
		self.delay[0]+=500*((np.abs(a1-self.pose.angle1)*180.0/np.pi)//5)#CHANGE IF CHANGED IN ARDUNIO
		self.delay[1]+=500*((np.abs(a2-self.pose.angle2)*180.0/np.pi)//5)#CHANGE IF CHANGED IN ARDUNIO
		self.pose.angle1=a1
		self.pose.angle2=a2
		
	def setHandUp(self,upq):
		self.pose.hand_up=upq
		self.delay[2]=2000
	def setHandOpen(self,openq):
		self.pose.hand_open=openq
		self.delay[3]=2000
		
	def getAndClearDelaySecs(self):
		re=max(self.delay)/1000.0
		for i in range(4):
			self.delay[i]=0
		return re

class CommandSenderThread(threading.Thread):
	def setUp(self,robot):
		self.robot = robot
		self.cmds=[]
		try:
			self.ser = serial.Serial("/dev/ttyACM0",9600)#each is a list of things with
			# tupes meaning go to square, or chars 'u','d','o','c'
		except:
			print("NO USB")
			self.ser=None
	def sendPickUpPieceSq(self,sq):#(),'o','d','c','u' assumes alread up
		self.cmds.append([sq])
		self.cmds.append(['o'])
		self.cmds.append(['d'])
		self.cmds.append(['c'])
		self.cmds.append(['u'])
	def sendPutPieceSq(self,sq):#(),'o','d','c','u' assumes alread up
		self.cmds.append([sq])
		self.cmds.append(['d'])
		self.cmds.append(['o'])
		self.cmds.append(['u'])
		self.cmds.append(['c'])
	def run(self):
		end_wait_time = time.time()
		while(True):
			time.sleep(.01)#10 ms
			if(len(self.cmds)>0 and time.time()>end_wait_time):
				#then do next command
				cmd = self.cmds[0]
				print("sending cmd:"+str(cmd))
				del self.cmds[0]
				for cm in cmd:
					if(cm=='u'):
						self.robot.setHandUp(True)
					elif(cm=='d'):
						self.robot.setHandUp(False)
					elif(cm=='o'):
						self.robot.setHandOpen(True)
					elif(cm=='c'):
						self.robot.setHandOpen(False)
					else:
						self.robot.set_angles_to_sq(cm)
				self.robot.sendState(self.ser)
				end_wait_time = time.time()+self.robot.getAndClearDelaySecs()
	
def startAndGetCMDThread(robot,id=1):				
	commandSender = CommandSenderThread(name = "Thread-{}".format(id))
	# ...Instantiate a thread and pass a unique ID to it
	commandSender.setUp(robot)
	commandSender.start() # ...Start the thread  
	return commandSender

	
	
	