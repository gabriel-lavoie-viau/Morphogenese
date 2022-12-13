from PIL import Image
import cv2
import numpy as np
import argparse

from face_detection import *
from Model import *
from Face_interpolation import *
from Camera_stream import *
from OSC_server import *


def process(monitor=0, tweaking_mode=False):
	# GLOBAL PARAMETERS

	tweak_mode 					= tweaking_mode
	rotate  					= False
	presentation_monitor		= monitor * 2000
	screen_width 				= 1920
	screen_height 				= 1080
	face_resolution				= (1024, 1024)
	crop_x						= 0.04 # Fraction of the image to crop on each side
	mask_is_white				= False
	camera_index 				= 0
	frame_rate 					= 20
	frame_interval 				= int(1000 / frame_rate)
	face_detection_interval 	= round(frame_rate * 0.5)
	flip_horizontally			= True


	# INSTANTIATION

	face_detector 				= face_detection(min_confidence=0.4, min_size=0.23, max_size=0.7, min_pos_x=0.25, max_pos_x=0.75)
	model       				= Model(face_resolution)
	# model.resize_dims			= (face_resolution)
	face_interpolator			= Face_interpolation(model, return_all_faces=tweak_mode)
	face_interpolator.start()
	osc_server         			= OSC_server(face_interpolator)
	osc_server.start()


	# BACKGROUND

	background = np.zeros((screen_height, screen_width, 3), np.uint8)
	if mask_is_white:
		background = np.uint8(background + 255.0)


	# MASK

	mask_amount = osc_server.mask_amount
	blur_amount = osc_server.blur_amount

	H, W 				= face_resolution[0], face_resolution[1]
	center_x, center_y 	= int(W*0.5), int(H*0.5 + H*0.05)

	mask_w, mask_h 				= int(W*0.18), int(H*0.3)
	mask_blur_w, mask_blur_h	= int(W*0.12), int(H*0.2)

	mask 		= np.zeros((H, W), np.uint8)
	mask_blur 	= np.zeros((H, W), np.uint8)

	cv2.ellipse(mask, (center_x, center_y), (mask_w, mask_h), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
	cv2.ellipse(mask_blur, (center_x, center_y), (mask_blur_w, mask_blur_h), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
	
	mask 		= cv2.blur(mask, (350, 350))
	mask_blur 	= cv2.blur(mask_blur, (350, 350))
	mask 		= cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
	mask_blur 	= cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR) / 255.0
	white_mask	= (1-mask) * 255.0
	# mask = cv2.GaussianBlur(mask, (41,41), 22)

	# CAMERA 

	cap = cv2.VideoCapture(camera_index)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920) 	# 640 || 1920
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080) # 320 || 1080
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))


	# PREPARE FOR PROCESS LOOP

	captured, img_cam = cap.read()
	if captured:
		# If able to get image from camera
		img_detect = img_cam.copy()
		frame = 0 
		previous_time = 0


		# MAIN WINDOW

		# Window name
		display = 'main_display'
		cv2.namedWindow(display, cv2.WND_PROP_FULLSCREEN)
		# Move the named windows accross screens
		cv2.moveWindow(display, presentation_monitor, 0) # Name, move x, move y
		# Make the named window fullscreen
		cv2.setWindowProperty(display,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


		# PROCESS LOOP

		while True:
			# Increment frame
			frame += 1

			# Read the camera image
			captured, img_cam = cap.read()

			# Detect faces
			if frame >= face_detection_interval:
				if img_cam is not None:
					img_detect, cropped_faces, num_user, interaction_time = face_detector.process(img_cam, draw=True, crop=True)
					
					# Save the cropped faces
					for face in cropped_faces:
						if face.any():
							index = cropped_faces.index(face)
							filepath = f'./img/cropped_faces/face_{index}.jpg'
							cv2.imwrite(filepath, face)

					# Send num_user and interaction time through OSC server
					osc_server.osc_send('/num_user', num_user)
					osc_server.osc_send('/interaction_time', interaction_time)

				frame = 0

			#Face generation process
			if tweak_mode:
				# If in tweak mode, get and display all the faces returned by the face_interpolator
				faces = face_interpolator.get_results()
				for face in range(len(faces)):
					faces[face] = np.array(faces[face])
					# Convert RGB to BGR
					if len(faces[face].shape) > 0:
						faces[face] = faces[face][:, :, ::-1].copy()
				
				faces[1] = cv2.resize(faces[1], (int(faces[1].shape[0]/4), int(faces[1].shape[1]/4)), interpolation=cv2.INTER_LINEAR)
				faces[3] = cv2.resize(faces[3], (int(faces[3].shape[0]/4), int(faces[3].shape[1]/4)), interpolation=cv2.INTER_LINEAR)
				faces[4] = cv2.resize(faces[4], (int(faces[4].shape[0]/4), int(faces[4].shape[1]/4)), interpolation=cv2.INTER_LINEAR)
				faces[6] = cv2.resize(faces[6], (int(faces[6].shape[0]/4), int(faces[6].shape[1]/4)), interpolation=cv2.INTER_LINEAR)		
				faces[2] = cv2.resize(faces[2], (int(faces[2].shape[0]/2), int(faces[2].shape[1]/2)), interpolation=cv2.INTER_LINEAR)
				faces[5] = cv2.resize(faces[5], (int(faces[5].shape[0]/2), int(faces[5].shape[1]/2)), interpolation=cv2.INTER_LINEAR)

				img_1 = cv2.hconcat([faces[1], faces[3], faces[4], faces[6]])
				img_2 = cv2.hconcat([faces[2], faces[5]])
				img_3 = cv2.vconcat([img_1, img_2, faces[0]])

				img = img_3

				# Draw a border around the image
				border_thickness = 3
				img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255, 255), border_thickness)

			else:
				# If not in tweak mode, get and display only the final face
				final_face = face_interpolator.get_results()
				final_face = np.array(final_face)
				# Convert RGB to BGR
				if len(final_face.shape) > 0:
					final_face = final_face[:, :, ::-1].copy()

				img = final_face
				img_blur = cv2.blur(img, (25, 25))

				blur_amount = osc_server.blur_amount
				alpha_blur = mask_blur + (1-blur_amount)
				alpha_blur = np.clip(alpha_blur, 0, 1)
				img = np.uint8(img*alpha_blur + img_blur*(1-alpha_blur))

				if mask_is_white:
					mask_amount = osc_server.mask_amount
					white_mask_alpha = white_mask * mask_amount
					img = np.uint8(np.clip(img + white_mask_alpha, 0, 255))
				else:
					mask_amount = osc_server.mask_amount
					alpha_mask = mask + (1-mask_amount)
					alpha_mask = np.clip(alpha_mask, 0 , 1)
					img = np.uint8(img * alpha_mask)

				# Cropping the image
				img = img[0:face_resolution[0], int(crop_x*face_resolution[1]):face_resolution[1]-int(crop_x*face_resolution[1])]


			# Get the frame per seconds
			current_time = time.time()
			fps = 1 / (current_time - previous_time)
			previous_time = current_time

			if rotate:
				cv2.putText(img, f'FPS: {round(fps, 2)}', (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
				img = np.rot90(img, 1)
				img_height, img_width, _ = img.shape
				y_offset = round((screen_height - img_height) / 2)
				x_offset = round((screen_width - img_width) / 2)
				img_final = background.copy()
				img_final[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img
			else:
				img = cv2.resize(img, (screen_height-int(screen_height*crop_x*2), screen_height), interpolation= cv2.INTER_LINEAR)
				img_height, img_width, _ = img.shape
				y_offset = round((screen_height - img_height) / 2)
				x_offset = round((screen_width - img_width) / 2)
				img_final = background.copy()
				img_final[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img
				# img_final = img
				# cv2.putText(img_final, f'FPS: {round(fps, 2)}', (10, img.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

			if osc_server.state == 0 and osc_server.eyes_are_open == False:
				text_luminosity = (1 - osc_server.led_ring_luminosity) * 255
				text_luminosity = (text_luminosity, text_luminosity, text_luminosity)

				text1 = 'PLACEZ-VOUS SUR LA MARQUE AU SOL'
				(text1_W, text1_H), text1_baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_PLAIN, 2, 2)
				text1_posx = int((img_final.shape[1] / 2) - (text1_W / 2))
				text1_posy = int(img_final.shape[0] / 4)
				cv2.putText(img_final, text1, (text1_posx, text1_posy), cv2.FONT_HERSHEY_PLAIN, 2, text_luminosity, 2)

				text2 = 'ET ENLEVEZ VOTRE MASQUE'
				(text2_W, text2_H), text2_baseline = cv2.getTextSize(text2, cv2.FONT_HERSHEY_PLAIN, 2, 2)
				text2_posx = int((img_final.shape[1] / 2) - (text2_W / 2))
				text2_posy = int(img_final.shape[0] / 4 + text2_H + 20)
				cv2.putText(img_final, text2, (text2_posx, text2_posy), cv2.FONT_HERSHEY_PLAIN, 2, text_luminosity, 2)

			if osc_server.state == 1 and osc_server.eyes_are_open == True:
				text_luminosity = 200
				text_luminosity = (text_luminosity, text_luminosity, text_luminosity)

				text1 = 'FAITES DES SONS'
				(text1_W, text1_H), text1_baseline = cv2.getTextSize(text1, cv2.FONT_HERSHEY_PLAIN, 2, 2)
				text1_posx = int((img_final.shape[1] / 2) - (text1_W / 2))
				text1_posy = int(img_final.shape[0] / 4)
				cv2.putText(img_final, text1, (text1_posx, text1_posy), cv2.FONT_HERSHEY_PLAIN, 2, text_luminosity, 2)

			if flip_horizontally:
				img_final = cv2.flip(img_final, 1)
				
			# Display the final image
			cv2.imshow(display, img_final)

			if tweak_mode:
				# If in tweak mode, create a window and display the camera and face detection images
				cam_final = cv2.hconcat([img_cam, img_detect])
				cam_final = cv2.resize(cam_final, (1920, 540), interpolation=cv2.INTER_LINEAR)
				if flip_horizontally:
					cam_final = cv2.flip(cam_final, 1)
				cv2.imshow('Camera', cam_final)

			# If escape key is pressed, stop process loop and other processes
			k = cv2.waitKey(frame_interval) & 0xff
			if k==27:
				face_interpolator.stop()
				cap.release()
				cv2.destroyAllWindows()
				break
	else:
		# If not able to get image from camera, print error message and stop other processes
		print('Unable to get image from camera at index', camera_index)
		face_interpolator.stop()
		cap.release()
		cv2.destroyAllWindows()

# def rotate_image(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result

# def horizontal_concatenate(img_list, interpolation=cv2.INTER_CUBIC):
#     h_min = min(img.shape[0] for img in img_list)
#     img_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation)
#                       for img in img_list]
#     return cv2.hconcat(img_list_resize)

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument('--monitor', type=int, default=0)
	parser.add_argument('--tweak_mode', type=int, default=0)

	args = parser.parse_args()
	
	monitor = args.monitor

	if args.tweak_mode:
		process(monitor, tweaking_mode=True)
	else:
		process(monitor, tweaking_mode=False)

if __name__ == '__main__':
	main()