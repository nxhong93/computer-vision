import numpy as np
import cv2
import argparse
import time
from imutils.object_detection import non_max_suppression

def detect_video(video, model):
	video_input = cv2.VideoCapture(video)

	while True:
		ret, frame = video_input.read()
		if ret == False:
			break


		orig = frame
		(h,w) = frame.shape[:2]

		# setting new width and height
		(newW, newH) = (320,320)
		rW = w/float(newW)
		rH = h/float(newH)

		#resize the image
		frame = cv2.resize(frame, (newW, newH))
		(h,w) = frame.shape[:2]

		# we define the two output layer names for the EAST Detector model
		# that we are interested in
		layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

		# loading pre-trained EAST detector
		net = cv2.dnn.readNet(model)

		blob = cv2.dnn.blobFromImage(frame, 1.0, (w,h), (123.68, 116.78, 103.94), swapRB = True, crop = False)

		net.setInput(blob)
		(scores, geometry) = net.forward(layers)

		# grab the rows and columns from score volume
		(numrows, numcols) = scores.shape[2:4]
		rects = [] #stores the bounding box coordiantes for text regions
		confidences = [] # stores the probability associated with each bounding box region in rects

		for y in range(0, numrows):
			scoresdata = scores[0,0,y]
			xdata0 = geometry[0,0,y]
			xdata1 = geometry[0,1,y]
			xdata2 = geometry[0,2,y]
			xdata3 = geometry[0,3,y]
			anglesdata = geometry[0,4,y]

			for x in range(0, numcols):
				if scoresdata[x]<0.1: # if score is less than min_confidence, ignore
					continue
			
				(offsetx, offsety) = (x*4.0, y*4.0) # EAST detector automatically reduces volume size as it passes through the network
				#extracting the rotation angle for the prediction and computing their sine and cos

				angle = anglesdata[x]
				cos = np.cos(angle)
				sin = np.sin(angle)

				h = xdata0[x] + xdata2[x]
				w = xdata1[x] + xdata3[x]
				

				endx = int(offsetx + (cos * xdata1[x]) + (sin * xdata2[x]))
				endy = int(offsety + (sin * xdata1[x]) + (cos * xdata2[x]))
				startx = int(endx - w)
				starty = int(endy - h)

				# appending the confidence score and probabilities to list
				rects.append((startx, starty, endx, endy))
				confidences.append(scoresdata[x])

		# applying non-maxima suppression to supppress weak and overlapping bounding boxes
		boxes = non_max_suppression(np.array(rects), probs = confidences)

		for(startx, starty, endx, endy) in boxes:
			startx = int(startx * rW)
			starty = int(starty * rH)
			endx = int(endx * rW)
			endy = int(endy * rH)


			cv2.rectangle(orig, (startx, starty), (endx, endy), (0,255,0), 2)

		cv2.imshow("text Detection", orig)
		if cv2.waitKey(1) & 0xFF == 27:
			break

detect_video(0, './pre_train/frozen_east_text_detection.pb')	