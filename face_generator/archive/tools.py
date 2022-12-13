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

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(1, parent_dir)

from models.psp import pSp 
from editings import latent_editor
from utils.alignment import align_face
from utils.common import tensor2im

''''''
'''MODEL'''
''''''
# DEFINE INFERENCE PARAMETERS
MODEL_ARGS = {
        "model_path": os.path.join(parent_dir, "pretrained_models/e4e_ffhq_encode.pt"),
        "image_path": os.path.join(parent_dir, "notebooks/images/input_img.jpg")
        }

# Setup required image transformations
MODEL_ARGS['transform'] = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# LOAD PRETRAINED MODEL
model_path = MODEL_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
pprint.pprint(opts)  # Display full options used

# update the training options
opts['checkpoint_path'] = model_path
# if 'learn_in_w' not in opts:
#     opts['learn_in_w'] = False
# if 'output_size' not in opts:
#     opts['output_size'] = 1024
opts= Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

''''''
'''INSTANTIATION AND GLOBALS'''
''''''
editor = latent_editor.LatentEditor(net.decoder, False)
ganspace_pca = torch.load(os.path.join(parent_dir, 'editings/ganspace_pca/ffhq_pca.pt'))

center_latents = torch.from_numpy(np.zeros((18, 512)).astype('float32')).unsqueeze(0).to("cuda")
encoded_latents = center_latents

resize_dims = (512, 512)

''''''
'''UTILITIES'''
''''''
def align_image(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

def apply_directions(directions, latents=encoded_latents):
    edited_latents = latents

    for direction in directions:
        if directions[direction][3] != 0:
            edited_latents = editor.apply_ganspace_custom(edited_latents, ganspace_pca, [directions[direction]])

    return edited_latents

def __run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents

def encode_image(image):
    global encoded_latents

    img_transforms = MODEL_ARGS['transform']
    transformed_image = img_transforms(image)

    with torch.no_grad():
        tic = time.time()
        images, encoded_latents = __run_on_batch(transformed_image.unsqueeze(0), net)
        encoded_image = tensor2im(images[0])
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
    
    return encoded_image, encoded_latents

def get_center_latents():
    return center_latents

def get_directions_list():
    # Load json file, set strengh to 0, convert values list to tuple
    with open('directions.json') as json_file:
        json_content = json.load(json_file)
        directions = json_content['available_directions']
        for direction in directions:
            directions[direction][3] = 0
            directions[direction] = tuple(directions[direction])

    return directions

def get_encoded_latents():
    return encoded_latents
    
def image_from_latents(latents):
    edited_image = editor.latents_to_image_custom(latents)
    resized_image = edited_image.resize(resize_dims)
    return resized_image

def latents_interpolation(scalar, latents):
    interpolated_latents = torch.lerp(latents, encoded_latents, scalar)
    return interpolated_latents

def random_latents():
    vec_to_inject = np.random.randn(1, 512).astype('float32') 
    _, latents = net(torch.from_numpy(vec_to_inject).to("cuda"), 
                            input_code=True, 
                            return_latents=True)
    
    return latents

def save_image(latents):
    edited_image = editor.latents_to_image_custom(latents)
    filename = "./exported_images/export_%i.png" % random.randrange(9999)
    edited_image.save(filename)