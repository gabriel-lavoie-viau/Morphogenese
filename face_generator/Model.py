import os
import sys
import time
from argparse import Namespace
import pprint
import json
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import dlib
import gc

current_dir = os.getcwd()
e4e_dir = os.path.join(current_dir, 'encoder4editing')
sys.path.insert(1, e4e_dir)


from models.psp import pSp 
from editings import latent_editor
from utils.alignment import align_face

from threading import Thread

gc.collect()
torch.cuda.empty_cache()


class Model:

    def __init__(self, resize_dims):

        # print(torch.cuda.is_available())
        # print(torch.cuda.current_device())
        # print(torch.cuda.device_count())
        print("Device Name :", torch.cuda.get_device_name(0))
        # print("Memory Sumary :",torch.cuda.memory_summary())

        # print("Memory Allocated", torch.cuda.memory_allocated())
        # print("Max Memory Allocated", torch.cuda.max_memory_allocated())
        # print("Memory Reserved", torch.cuda.memory_reserved())
        # print("Max Memory Reserved", torch.cuda.max_memory_reserved())

        ''''''
        '''MODEL'''
        ''''''
        self.resize_dims = resize_dims # (256, 256)

        # DEFINE INFERENCE PARAMETERS
        self.MODEL_ARGS = {
                "model_path": './encoder4editing/pretrained_models/e4e_ffhq_encode.pt',
                "image_path": './inputs/input_img.jpg'
                }

        # Setup required image transformations
        self.MODEL_ARGS['transform'] = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize(self.resize_dims),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # LOAD PRETRAINED MODEL
        model_path = self.MODEL_ARGS['model_path']
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        # pprint.pprint(opts)  # Display full options used

        # update the training options
        opts['checkpoint_path'] = model_path
        # if 'learn_in_w' not in opts:
        #     opts['learn_in_w'] = False
        # if 'output_size' not in opts:
        #     opts['output_size'] = 1024
        opts= Namespace(**opts)
        self.net = pSp(opts)
        self.net.eval()
        self.net.cuda()
        print('Model successfully loaded!')

        ''''''
        '''INSTANTIATION AND GLOBALS'''
        ''''''
        self.editor = latent_editor.LatentEditor(self.net.decoder, False)
        self.ganspace_pca = torch.load(os.path.join(e4e_dir, 'editings/ganspace_pca/ffhq_pca.pt'))

        _, self.center_latents = self.net(torch.from_numpy(np.zeros((1, 512)).astype('float32')).to("cuda"), input_code=True, return_latents=True)
        self.encoded_latents = self.center_latents


    ''''''
    '''UTILITIES'''
    ''''''
    def align_image(self, image_path):
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(filepath=image_path, predictor=predictor) 
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image 

    def apply_directions(self, directions, latents=None):
        if latents == None:
            latents = self.encoded_latents

        edited_latents = latents

        for direction in directions:
            if directions[direction][3] != 0:
                edited_latents = self.editor.apply_ganspace_custom(edited_latents, self.ganspace_pca, [directions[direction]])

        return edited_latents

    def get_center_latents(self):
        return self.center_latents

    def get_directions_list(self):
        # Load json file, set strengh to 0, convert values list to tuple
        with open('directions.json') as json_file:
            json_content = json.load(json_file)
            directions = json_content['available_directions']

        return directions

    def get_encoded_latents(self):
        return self.encoded_latents
        
    def image_from_latents(self, latents):
        # print("Memory Sumary :",torch.cuda.memory_summary())

        edited_image = self.editor.latents_to_image_custom(latents)
        resized_image = edited_image.resize(self.resize_dims)
        
        return resized_image

    def __run_on_batch(self, inputs):
        # images, latents = self.net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True, resize=False)
        images, latents = self.net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
        return images, latents

    def latents_from_image(self, image):
        img_transforms = self.MODEL_ARGS['transform']
        transformed_image = img_transforms(image)

        # print(img_transforms)
        # print(transformed_image)

        with torch.no_grad():
            tic = time.time()
            _, self.encoded_latents = self.__run_on_batch(transformed_image.unsqueeze(0))

            # self.image_from_latents(self.encoded_latents).show()

            toc = time.time()
            print('Inference took {:.4f} seconds.'.format(toc - tic))
        
        return self.encoded_latents

    def latents_interpolation(self, latents_1, latents_2, scalar):
        # print("Memory Allocated", torch.cuda.memory_allocated())
        # print("Max Memory Allocated", torch.cuda.max_memory_allocated())
        # print("Memory Reserved", torch.cuda.memory_reserved())
        # print("Max Memory Reserved", torch.cuda.max_memory_reserved())

        interpolated_latents = torch.lerp(latents_1, latents_2, scalar)
        return interpolated_latents

    def random_latents(self):
        vec_to_inject = np.random.randn(1, 512).astype('float32') 
        _, latents = self.net(torch.from_numpy(vec_to_inject).to("cuda"), 
                                input_code=True, 
                                return_latents=True)
        
        return latents

    def save_image(self, latents):
        edited_image = self.editor.latents_to_image_custom(latents)
        filename = "./exported_images/export_%i.png" % random.randrange(9999)
        edited_image.save(filename)