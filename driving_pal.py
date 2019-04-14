# import the necessary packages
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
distract_model = load_model('distraction_model.hdf5', compile=False)

# frame params
frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5

def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[5], mouth[8])
    B = dist.euclidean(mouth[1], mouth[11])	
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C) 	
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
MOUTH_AR_THRESH=0.20	
MOUTH_AR_CONSEC_FRAMES = 24
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
MOUTH_COUNTER =0 
ALARM_ON = False
WARN =False;

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#video_out = cv2.VideoWriter('video_out.avi', fourcc, 10.0,(1200, 900))

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(0);
time.sleep(1.0)

if (vs.isOpened()==False) :
	print("Unable to read camera feed")
# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	ret,frame = vs.read()
	frame = imutils.resize(frame, width=1200)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		jaw= shape[48:61];
		
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		mar=mouth_aspect_ratio(jaw)/ 2.0;
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		jawHull= cv2.convexHull(jaw)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
 		

 		if mar > MOUTH_AR_THRESH:
			MOUTH_COUNTER +=1;
			
			if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
					if not WARN:
						WARN = True;
					cv2.putText(frame, "YOU ARE YAWNING", (10, 50), # first the horizontal distance then y
						cv2.FONT_HERSHEY_SIMPLEX, 2, (0,100,0), 3)
					cv2.putText(frame, "STOP FOR A COFFEE", (10, 95),
						cv2.FONT_HERSHEY_SIMPLEX, 2, (0,100,0), 3)	
		else :
			MOUTH_COUNTER =0
			WARN = False 
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (1000, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (153, 76, 0), 2)


		if WARN == False:
			if ear < EYE_AR_THRESH:
				COUNTER += 1

				# if the eyes were closed for a sufficient number of
				# then sound the alarm
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					# if the alarm is not on, turn it on
					if not ALARM_ON:
						ALARM_ON = True

						# check to see if an alarm file was supplied,
						# and if so, start a thread to have the alarm
						# sound played in the background
						if args["alarm"] != "":
							t = Thread(target=sound_alarm,
								args=(args["alarm"],))
							t.deamon = True
							t.start()

					# draw an alarm on the frame
					cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),
						cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

			# otherwise, the eye aspect ratio is not below the blink
			# threshold, so reset the counter and alarm
			else:
				COUNTER = 0
				ALARM_ON = False
			
		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (1000, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1, (153, 76, 0), 2)
	
	
	#Distraction detector

 	faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,
 		minSize=(min_size_w,min_size_h),flags=cv2.CASCADE_SCALE_IMAGE)
 	#print(faces)	
 	
            # loop through faces
        if len(faces)>0 :   
		
		for (x,y,w,h) in faces:
		        # draw face rectangle

		        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		        # get gray face for eye detection
		        roi_gray = gray[y:y+h, x:x+w]
		        # get colour face for distraction detection (model has 3 input channels - probably redundant)
		        roi_color = frame[y:y+h, x:x+w]
		        # detect gray eyes
		        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize
		        =(min_size_w_eye,min_size_w_eye))

		        # init probability list for each eye prediction
		        probs = list()

		        # loop through detected eyes
		        for (ex,ey,ew,eh) in eyes:
		            # draw eye rectangles
		            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
		            # get colour eye for distraction detection
		            roi = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]
		            # match CNN input shape
		            roi = cv2.resize(roi, (64, 64))
		            # normalize (as done in model training)
		            roi = roi.astype("float") / 255.0
		            # change to array
		            roi = img_to_array(roi)
		            # correct shape
		            roi = np.expand_dims(roi, axis=0)

		            # distraction classification/detection
		            prediction = distract_model.predict(roi)
		            # save eye result
		            probs.append(prediction[0])

		        # get average score for all eyes
		        probs_mean = np.mean(probs)

		        # get label
		        label='focused'
		        if probs_mean <= 0.3:		# 0.5 doesn't work in practical situations
		            label = 'distracted'
		            cv2.putText(frame,"PLEASE FOCUS ON DRIVING",(200,895), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 4)    
		        else:
			    label = 'focused'
			
			if label=='focused':
				cv2.putText(frame,label,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
			elif label=='distracted':
				cv2.putText(frame,label,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)	
	
 	#video_out.write(frame)	 		
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()		
