import cv2
import mediapipe as mp
import time



class face_detection():
	def __init__(self, min_confidence=0.5, min_size=0.1, max_size=0.9, min_pos_x=0.25, max_pos_x=0.75):

		self.min_confidence	= min_confidence
		self.min_size		= min_size
		self.max_size		= max_size
		self.min_pos_x		= min_pos_x
		self.max_pos_x		= max_pos_x

		# Initialize face detection class

		# model_selection
		# An integer index 0 or 1. Use 0 to select a short-range model that works best 
		# for faces within 2 meters from the camera, and 1 for a full-range model best 
		# for faces within 5 meters. For the full-range option, a sparse model is used 
		# for its improved inference speed. Please refer to the model cards for details. 
		# Default to 0 if not specified.
		
		# min_detection_confidence
		# Minimum confidence value ([0.0, 1.0]) from the face detection model for the 
		# detection to be considered successful. Default to 0.5.

		self.face_detection = mp.solutions.face_detection.FaceDetection(self.min_confidence)
		self.mp_draw 		= mp.solutions.drawing_utils
		self.previous_num_faces = 0
		self.interaction_start = 0

	def find_faces(self, img, draw=False):

		img_detect = img.copy()
		# Convert to RGB
		img_RGB = cv2.cvtColor(img_detect, cv2.COLOR_BGR2RGB)
		# Detect the faces
		faces = self.face_detection.process(img_RGB)
		# Reset bounding_box_list
		bounding_box_list = []

		if faces.detections:
			for id, detection in enumerate(faces.detections):
				# print(id, detection)
				bbox_mp = detection.location_data.relative_bounding_box
				position_x = bbox_mp.xmin + (bbox_mp.width / 2)

				if bbox_mp.height > self.min_size and bbox_mp.height < self.max_size:
					
					if position_x > self.min_pos_x and position_x < self.max_pos_x:
						height, width, channels = img_detect.shape
						bounding_box = 	int(bbox_mp.xmin * width), int(bbox_mp.ymin * height), \
										int(bbox_mp.width * width), int(bbox_mp.height * height)
						if draw == True:
							cv2.rectangle(img_detect, bounding_box, (255, 255, 255), 1)
							cv2.putText(img_detect, f'Confidence: {int(detection.score[0] * 100)}%', 
										(bounding_box[0], bounding_box[1] - 10),
										cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
							cv2.putText(img_detect, f'Size: {int(bbox_mp.height * 100)}%', 
										(bounding_box[0], bounding_box[1] + bounding_box[3] + 20),
										cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
							cv2.putText(img_detect, f'Position: {int(position_x * 100)}%', 
										(bounding_box[0], bounding_box[1] + bounding_box[3] + 40),
										cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

						bounding_box_list.append([bounding_box, detection.score])

		return img_detect, bounding_box_list


	def crop_image(self, img, bounding_box_list):
		
		cropped_faces = []
		padding = int(img.shape[0] * 0.2)
		
		for bbox in bounding_box_list:
			X, Y, W, H = 	bbox[0][0] - padding	, bbox[0][1] - padding, \
							bbox[0][2] + (padding*2), bbox[0][3] + (padding*2)

			Y = 0 if Y < 0 else Y
			X = 0 if X < 0 else X
			W = img.shape[1] if W > img.shape[1] else W
			H = img.shape[0] if H > img.shape[0] else H

			face = img[Y:Y+H, X:X+W]

			cropped_faces.append(face)

		return cropped_faces


	def process(self, img, draw=False, crop=False):

		img_detect, bounding_box_list = self.find_faces(img, draw=draw)

		num_faces = len(bounding_box_list)

		if num_faces > 0 and self.previous_num_faces == 0:
				self.interaction_start = time.time()
		if num_faces == 0:
				self.interaction_start = time.time()
		
		self.previous_num_faces = num_faces
		interaction_time = time.time() - self.interaction_start

		if crop == True:
			cropped_faces = self.crop_image(img, bounding_box_list)

		if draw:
			# Show user infos
			cv2.putText(img_detect, f'Num faces: {num_faces}', (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
			cv2.putText(img_detect, f'Interaction time: {round(interaction_time, 2)}', (10,40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


		return img_detect, cropped_faces, num_faces, interaction_time