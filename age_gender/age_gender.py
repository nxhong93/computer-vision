import numpy as np
import sys
import cv2
from centroidtracker import CentroidTracker


path_face = "C:/ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
age_txt = './pre_train/deploy_age.prototxt'
age_model = './pre_train/age_net.caffemodel'
gender_txt = './pre_train/deploy_gender.prototxt'
gender_model = './pre_train/gender_net.caffemodel'
txt = './pre_train/deploy.prototxt.txt'
model_ = './pre_train/res10_300x300_ssd_iter_140000.caffemodel'
video = './video/chaplin.mp4'
ct = CentroidTracker()

# def detect_face(frame, path_face):
# 	#load path face
# 	face_ = cv2.CascadeClassifier(path_face)
# 	#convert frame to gray
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	#detect face
# 	detect_face = face_.detectMultiScale(gray, 1.1, 5)
# 	return detect_face

def detect_face(frame, txt, model):
	#load model cafe
	net = cv2.dnn.readNetFromCaffe(txt, model)
	#add input frame to model
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
	net.setInput(blob)
	#detect object
	blobs = net.forward()
	
	return blobs

def detect_age(box, age_txt, age_model):
	MODEL_MEAN_VALUES = (78, 87, 114)
	#age list
	ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
	#load model
	ageNet = cv2.dnn.readNetFromCaffe(age_txt, age_model)
	#input box
	blob = cv2.dnn.blobFromImage(box, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
	ageNet.setInput(blob)
	#detect age
	age_pred = ageNet.forward()
	print(age_pred)
	age = ageList[age_pred[0].argmax()]
	return age

def detect_gender(box, gender_txt, gender_model):
	MODEL_MEAN_VALUES = (78, 87, 114)
	#gender list
	genderList = ['Male', 'Female']
	#load model
	genderNet = cv2.dnn.readNetFromCaffe(gender_txt, gender_model)
	#input box
	blob = cv2.dnn.blobFromImage(bo
		while True:x, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
	genderNet.setInput(blob)
	#detect age
	gender_pred = genderNet.forward()
	gender = genderList[gender_pred[0].argmax()]
	return gender	
			
def main(video, threshold, padding):

	#input video
	video_input = cv2.VideoCapture(video)

	#check if or not have video
	if not video_input.isOpened():
		print('Not found video')
		sys.exit()
	else:
		box_id = 0
			ret, frame = video_input.read()

			if ret == False:
				print('video ended')
				break

			h, w = frame.shape[:2]

			rects = []			
			
			detect_faces = detect_face(frame, txt, model_)
			for box_number in range(detect_faces.shape[2]):
				confident = detect_faces[0,0,box_number,2]

				if confident > threshold:
					box = detect_faces[0,0,box_number,3:7]*np.array([w,h,w,h])
					box = (box.astype('int'))
					
					rects.append(box)
						
					cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2, 1)
					# cv2.rectangle(frame, (int(box[0]-10), int(box[1])-10), (int(box[2]+10), int(box[3]+10)), (255,255,0), 2, 1)

					face_image = frame[box[1]-padding:box[3]+padding, box[0]-padding:box[2]+padding]
					try:
						age = detect_age(face_image, age_txt, age_model)
						gender = detect_gender(face_image, gender_txt, gender_model)

						label = '{}{}'.format(gender, age)
						cv2.putText(frame, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
					except:
						print('error')
			try:
				objects = ct.update(rects)
				for objectId, center in objects.items():
					text_id = 'ID={}'.format(objectId)
					if objectId > box_id:
						box_id += 1
					cv2.putText(frame, text_id, (center[0]-10, center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
					cv2.circle(frame, (center[0], center[1]), 4, (255,0,0), -1)
			except:
				print('error')

			cv2.imshow('frame', frame)
			if cv2.waitKey(1) & 0xFF == 27:
				break
if __name__ == '__main__':
	main(0, 0.7, 15)
