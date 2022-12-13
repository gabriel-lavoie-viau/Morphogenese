from threading import Thread
from queue import Queue
from PIL import ImageTk #, Image
import numpy as np
import torch
import gc
import time
import json
import random


class Face_interpolation:

    def __init__(self, model=None, return_all_faces=False):
        
        self.model = model
        self.return_all_faces = return_all_faces

        self.saved_faces            = torch.load('saved_faces_bckup2.pt')
        self.max_saved_user_faces   = 1000
        self.max_rdm_user_faces     = 60

        # # ERASE ALL SAVED FACES EXCEPT #1 AND #2
        # saved_faces_temp = {'user': {'0': self.saved_faces['user']['0'], '1': self.saved_faces['user']['1']}}
        # print('\n\n', saved_faces_temp)
        # torch.save(self.saved_faces, 'saved_faces_bckup2.pt')
        # torch.save(saved_faces_temp, 'saved_faces.pt')

        self.user_vs_dataset_face_interpolation = 0.5

        self.user_face_interpolation_state  = 0.0
        self.user_face_interpolation_speed  = 0.008
        self.user_previous_face             = self.saved_faces['user']['1']
        self.user_present_face              = self.user_previous_face
        self.user_next_face                 = self.saved_faces['user']['0']
        # self.user_previous_face             = self.model.random_latents()
        # self.user_present_face              = self.user_previous_face
        # self.user_next_face                 = self.model.random_latents()
        self.user_previous_face_img         = self.model.image_from_latents(self.user_previous_face)
        self.user_present_face_img          = self.model.image_from_latents(self.user_present_face)
        self.user_next_face_img             = self.model.image_from_latents(self.user_next_face)

        self.dataset_face_interpolation_state   = 0.0
        self.dataset_face_interpolation_speed   = 0.008
        self.dataset_previous_face              = self.model.random_latents()
        self.dataset_present_face               = self.dataset_previous_face
        self.dataset_next_face                  = self.model.random_latents()
        self.dataset_previous_face_img          = self.model.image_from_latents(self.dataset_previous_face)
        self.dataset_present_face_img           = self.model.image_from_latents(self.dataset_present_face)
        self.dataset_next_face_img              = self.model.image_from_latents(self.dataset_next_face)

        self.jitter_iterpolation_state  = 0.0
        self.jitter_amplitude           = 0.2
        self.jitter_speed               = 0.03
        self.jitter_start               = self.user_present_face
        self.jitter_end                 = None

        self.directions = self.model.get_directions_list()

        self.final_face         = self.user_present_face
        self.final_face_img     = None

        self.is_on          = False
        self.process_thread = None


    def interpolate_user_face(self):
        if self.user_face_interpolation_state >= 1.0:
            return self.user_previous_face
        else:
            self.user_present_face = torch.lerp(self.user_previous_face, self.user_next_face, self.user_face_interpolation_state)
            self.user_face_interpolation_state += self.user_face_interpolation_speed

            if self.user_face_interpolation_state >= 1.0:
                self.user_previous_face = self.user_next_face

            return self.user_present_face


    def interpolate_dataset_face(self):
        if self.dataset_face_interpolation_state >= 1.0:
            return self.dataset_previous_face
        else:
            self.dataset_present_face = torch.lerp(self.dataset_previous_face, self.dataset_next_face, self.dataset_face_interpolation_state)
            self.dataset_face_interpolation_state += self.dataset_face_interpolation_speed

            if self.dataset_face_interpolation_state >= 1.0:
                self.dataset_previous_face = self.dataset_next_face

            return self.dataset_present_face


    def jitter(self):
        if self.jitter_iterpolation_state == 0.0:
            self.jitter_end = self.jitter_amplitude * torch.from_numpy(np.random.randn(18, 512).astype('float32')).unsqueeze(0).to("cuda")
        
        latents = torch.lerp(self.jitter_start, self.jitter_end, self.jitter_iterpolation_state)
        self.jitter_iterpolation_state += self.jitter_speed

        if self.jitter_iterpolation_state >= 1.0:
            self.jitter_start = self.jitter_end
            self.jitter_iterpolation_state = 0.0

        return latents


    def apply_directions(self, latents):

        latents = self.model.apply_directions(self.directions, latents)

        return latents


    def __process(self):
        while self.is_on:
            tic = time.time()

            user_face           = self.interpolate_user_face()     
            dataset_face        = self.interpolate_dataset_face()        
            self.final_face     = self.model.latents_interpolation(user_face, dataset_face, self.user_vs_dataset_face_interpolation)
            self.final_face     = self.final_face  + self.jitter()
            self.final_face     = self.apply_directions(self.final_face)          

            if self.return_all_faces == True: 
                self.user_present_face_img      = self.model.image_from_latents(user_face)
                self.dataset_present_face_img   = self.model.image_from_latents(dataset_face)
            self.final_face_img = self.model.image_from_latents(self.final_face)

            time.sleep(0.01)
            toc = time.time()

            # print('FPS : {:.2f}'.format(1/ (toc - tic)))


    def start(self):
        self.is_on = True
        self.process_thread = Thread(target=self.__process, args=())
        self.process_thread.start()


    def stop(self):
        self.is_on = False


    def get_results(self):
        if self.return_all_faces == True:
            return [self.final_face_img, 
                    self.user_previous_face_img, 
                    self.user_present_face_img, 
                    self.user_next_face_img, 
                    self.dataset_previous_face_img, 
                    self.dataset_present_face_img, 
                    self.dataset_next_face_img]
        else:
            return self.final_face_img
    

    def change_user_face(self, face_index=None):
        self.user_previous_face = self.user_present_face
        self.user_face_interpolation_state = 0.0
    
        face_num = len(self.saved_faces['user'])
        if face_index == None:
            rdm_face = random.randint(0, self.max_rdm_user_faces)
            self.user_next_face = self.saved_faces['user'][str(rdm_face)]
        else:
            if face_index < face_num:
                self.user_next_face = self.saved_faces['user'][str(face_index)]
            else:
                print("No face at that index")

        self.user_previous_face_img     = self.model.image_from_latents(self.user_previous_face)
        self.user_next_face_img         = self.model.image_from_latents(self.user_next_face)


    def change_dataset_face(self, face_index=None, generate_random=False):
        self.dataset_previous_face = self.dataset_present_face
        self.dataset_face_interpolation_state = 0.0

        face_num = len(self.saved_faces['dataset'])
        if face_index == None:
            if generate_random:
                self.dataset_next_face = self.model.random_latents()
            else:
                rdm_face = random.randint(0, face_num-1)
                self.dataset_next_face = self.saved_faces['dataset'][str(rdm_face)]
        else:
            if face_index < face_num:
                self.dataset_next_face = self.saved_faces['dataset'][str(face_index)]
            else:
                print("No face at that index")

        self.dataset_previous_face_img  = self.model.image_from_latents(self.dataset_previous_face)
        self.dataset_next_face_img      = self.model.image_from_latents(self.dataset_next_face)


    def encode_face(self, path_to_img="./img/cropped_faces/face_0.jpg"):
        aligned_image = self.model.align_image(path_to_img)
        aligned_image = aligned_image.resize((self.model.resize_dims))

        encoded_latents = self.model.latents_from_image(aligned_image)
        self.save_latents(is_user_face=True, latents=encoded_latents)

    def save_latents(self, is_user_face=True, latents=None):
        if is_user_face == True:
            if latents == None:
                print("No latents provided")
            else:
                face_num = len(self.saved_faces['user'])
                for num in range(face_num):
                    inc = face_num - num
                    if inc < self.max_saved_user_faces:
                        self.saved_faces['user'][str(inc)] = self.saved_faces['user'][str(inc-1)]
                self.saved_faces['user']['0'] = latents
                print("Saved user face #" + str(face_num) + " at position 0")
        else:
            face_num = str(len(self.saved_faces['dataset']))
            self.saved_faces['dataset'][face_num] = self.final_face
            print("Saved dataset face #" + str(face_num))
        
        torch.save(self.saved_faces, 'saved_faces.pt')


    def set_user_vs_dataset_face_interpolation(self, val=0.0):
        self.user_vs_dataset_face_interpolation = val


    def set_jitter_amplitude(self, amp=0.2):
        self.jitter_amplitude = amp


    def set_jitter_speed(self, speed=0.03):
        self.jitter_speed = speed


    def set_directions(self, direction, value):
        self.directions[direction][3] = value


    def save_directions(self, title):
        with open('directions.json') as json_file:
            json_content = json.load(json_file)
        with open('directions.json', 'w') as json_file:
            json_content[title] = self.directions
            json.dump(json_content, json_file)   
    

    def export_image(self):
        img = self.model.image_from_latents(self.final_face)
        filename = 'exported_images/image' + str(random.randint(0, 9999)) + '.jpg'
        img.save(filename)

    # def set_tk_panel(self, panel=None):
    #     self.tk_panel = panel

    # def get_final_face(self):
    #     return self.final_face  





    # def get_directions(self):
    #     return self.directions


