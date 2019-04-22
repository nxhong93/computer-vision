import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import cv2
import imutils
from imutils.object_detection import non_max_suppression

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def main(image, threshold):

	#read image
	img = cv2.imread(image)
	h, w = img.shape[:2]
	
	#convert image to gray
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#create threshold for image with binary kind
	_,thresh_image = cv2.threshold(img_gray, threshold,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

	#kernel length
	kernel_length = w//50

	#vertical kernel
	ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

	#horizon kernel
	hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

	#kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

	#vertical lines
	vertical_temp = cv2.erode(thresh_image, ver_kernel, (-1,-1))
	vertical_lines = cv2.dilate(vertical_temp, ver_kernel, (-1,-1))

	#horizon lines
	horizon_temp = cv2.erode(thresh_image, hor_kernel, (-1,-1))
	horizon_lines = cv2.dilate(horizon_temp, hor_kernel, (-1,-1))

	#add vertical and horizon lines
	img_final = cv2.addWeighted(vertical_lines, 0.5, horizon_lines, 0.5, 0.0)
	# img_final = cv2.erode(~img_final, kernel, (-1,-1))
	_, img_final = cv2.threshold(img_final, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	

	#remove vertical and horizon lines
	horizontal_inv = cv2.bitwise_not(img_final)
	masked_img = cv2.bitwise_not(thresh_image, thresh_image, mask=horizontal_inv)	

	# minAreaRect on the nozeros
	pts = cv2.findNonZero(masked_img)
	ret = cv2.minAreaRect(pts)

	(cx,cy), (w_,h_), ang = ret
	if w_ < h_:
	    w_,h_ = h_,w_
	    ang += 90

	## Find rotated matrix, do rotation
	M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
	image_rotation = cv2.warpAffine(masked_img, M, (img.shape[1], img.shape[0]))

	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))
	image_text = cv2.dilate(image_rotation, rect_kernel, iterations = 1)
	#find counter
	_, counters, _ = cv2.findContours(image_text, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	
	text_box = []
	for counter in counters:		
		x, y, w, h = cv2.boundingRect(counter)
		
		if w>3*h and h>15:
			cv2.rectangle(image_rotation, (x, y), (x+w, y+h), (255,255,0), 2)

			image_crop = image_rotation[y:y+h, x:x+w]
			
			text = pytesseract.image_to_string(image_crop, lang='vietnamese')
			print(text)
			text_box.append(text)
			# cv2.imshow('image_roation', image_crop)
			# cv2.waitKey()

	cv2.imshow('image_rotation', image_rotation)
	cv2.waitKey()	

	
if __name__ == '__main__':
	main('./anh/2.png', 180)
	
