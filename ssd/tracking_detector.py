import cv2
import sys
from imutils.video import VideoStream, FPS
import imutils
import numpy as np

def tracking_detector(video, tracker_type, prototxt, model):
	
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	net = cv2.dnn.readNetFromCaffe(prototxt, model)

	tracker_multiply = cv2.MultiTracker_create()
	def track_t(tracker_):

		tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']

		if tracker_.upper() not in tracker_types:
			tracker_index = None
			
		else:
			tracker_index = tracker_types.index(tracker_.upper())

		if tracker_index == 0:
			tracker = cv2.TrackerBoosting_create()
		elif tracker_index == 1:
			tracker = cv2.TrackerMIL_create()
		elif tracker_index == 2:
			tracker = cv2.TrackerKCF_create()
		elif tracker_index == 3:
			tracker = cv2.TrackerTLD_create()
		elif tracker_index == 4:
			tracker = cv2.TrackerMedianFlow_create()
		elif tracker_index == 5:
			tracker = cv2.TrackerCSRT_create()
		elif tracker_index == 6:
			tracker = cv2.TrackerMOSSE_create()
		else:
			tracker = None
			print('Not found tracker type')
			sys.exit()
		return tracker

	
	#read video
	video_input = cv2.VideoCapture(video)

	#check if or not video
	if not video_input.isOpened():
		print('Not found video')
		sys.exit()
	else:
		while True:
			ret, frame = video_input.read()

			if ret == False:
				break

			#resize help faster
			# frame = imutils.resize(frame, width=700)
			h, w = frame.shape[:2]
			
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

			# pass the blob through the network and obtain the detections and
			# predictions
			print("[INFO] computing object detections...")
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with the
				# prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence > 0.5:
					# extract the index of the class label from the `detections`,
					# then compute the (x, y)-coordinates of the bounding box for
					# the object
					idx = int(detections[0, 0, i, 1])
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")

					# display the prediction
					label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
					print("[INFO] {}".format(label))
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			#check if init_box is defined
			
			ok, boxes = tracker_multiply.update(frame)

			for box in boxes:
				start = (int(box[0]), int(box[1]))
				end = (int(box[0]+box[2]), int(box[1]+box[3]))

				cv2.rectangle(frame, start, end, (0,255,0), 2, 1)
				cv2.putText(frame, str(start), (start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
				cv2.putText(frame, str(end), (end), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
				


			# fps.update()
			# fps.stop()
		
			#display tracker type
			cv2.putText(frame, tracker_type + ' tracker', (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

			#display video
			cv2.imshow('frame', frame)

			#exis video if click button ESC
			if cv2.waitKey(1) & 0xFF == 27:
				cv2.destroyAllWindows()
				break
			elif cv2.waitKey(1) & 0xFF == ord('s'):
				box = cv2.selectROI('frame', frame, showCrosshair=True, fromCenter=False)

				tracker = track_t(tracker_type)
				tracker_multiply.add(tracker, frame, box)

			

			
prototxt = "./pre_train/MobileNetSSD_deploy.prototxt.txt"
model = "./pre_train/MobileNetSSD_deploy.caffemodel"
video = './video/chaplin.mp4'
tracking_detector(video, 'csrt', prototxt, model)				
