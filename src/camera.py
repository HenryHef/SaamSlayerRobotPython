# http://support.logitech.com/en_in/product/webcam-c170/specs
# Optical Resolution	True 640x480
# Diagonal Field of View (FOV)	58°
# Focal Length	2.3 mm
import numpy as np
from chessboard import chess_board_points
import cv2 as cv



class Camera:
	pix_dimens = (640,480)
	pix_diag = np.sqrt(pix_dimens[0]**2+pix_dimens[1]**2)
	diag_fov = 58*np.pi/180.0# in rad
	
	horez_fov = 2.0*np.arctan(pix_dimens[0]/pix_diag*np.tan(diag_fov/2.0))
	#focal_pixel = (pix_dimens[0] * 0.5) / np.tan(horez_fov * 0.5)
	
	focal_pixel = pix_diag*np.tan(diag_fov/2.0) * 0.5
	
	cameraMatrix=np.float32(
		[
			[focal_pixel  ,0			,pix_dimens[0]],
			[0			  ,focal_pixel  ,pix_dimens[1]],
			[0			  ,0			,1]
		])




