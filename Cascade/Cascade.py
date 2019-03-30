import numpy as np
import cv2

def video_cascade(video, object):
	path = "C:\\ProgramData\\Anaconda3\\Lib\\site-packages\\cv2\\data\\" + object
	object_ = cv2.CascadeClassifier(path)
	video_input = cv2.VideoCapture(video)
	ret = True
	while ret == True:
		ret, frame = video_input.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		object_detect = object_.detectMultiScale(gray, 1.3, 5)

		for x, y, w, h in object_detect:
			frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			text1, text2 = str((x,y)), str((x+w,y+h))
			cv2.putText(frame, text1, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
			cv2.putText(frame, text2, (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == 27:
			break



video_cascade('./video/CAMERA.mp4', 'haarcascade_frontalface_alt.xml')
