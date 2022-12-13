from threading import Thread
import math
from scipy.ndimage import interpolation
import numpy as np
from tensorflow.compat.v2 import convert_to_tensor
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
# from math import ceil


class OSC_server:

    def __init__(self, ip="127.0.0.1", port=8201, model=None):
        # OSC paramters 
        self.ip     = ip
        self.port   = port
        self.client = SimpleUDPClient("127.0.0.1", 8200)

        self.model = model

        num_voices      = 6
        self.voices     = {}

        for i in range(num_voices):
            voice_name = 'voice' + str(i)
            self.voices[voice_name] = {}

        for voice in self.voices:
            self.voices[voice]['user_loudness']         = []
            self.voices[voice]['dataset_loudness']      = []
            self.voices[voice]['loudness_extrapolate']  = 1.0
            self.voices[voice]['user_mfcc']             = np.empty(1)
            self.voices[voice]['dataset_mfcc']          = []
            self.voices[voice]['mfcc_extrapolate']      = 1.0
            self.voices[voice]['pitch_scaler']          = 1.0
            self.voices[voice]['loudness_scaler']       = 1.0

            self.voices[voice]['user_mfcc_list']        = []
            self.voices[voice]['dataset_mfcc_list']     = []

            self.voices[voice]['audio_features']        = {}

            # Create dummy audio feature before loading the model
            self.voices[voice]['audio_features'] = {'num_samples':      12800,
                                                    'loudness_db':      np.random.rand(200) * 120,
                                                    'f0_hz':            np.random.rand(200) * 800,
                                                    'f0_confidence':    np.full((200), 1.0),
                                                    'phoneme':          np.random.rand(200) * 50,
                                                    'audio':            None,
                                                    'mfccs':            convert_to_tensor(np.random.rand(1,30,30)*150)
                                                    }

        # Create dispatcher
        self.dispatcher = Dispatcher()
        # Create server
        self.server = BlockingOSCUDPServer((self.ip, self.port), self.dispatcher)
        # Create function for unknown messages
        self.dispatcher.set_default_handler(self.default_handler)

        # Calls to functions
        self.dispatcher.map("/load_model",          self.load_model)
        self.dispatcher.map("/predict",             self.predict)
        self.dispatcher.map("/num_samples",         self.num_samples)
        self.dispatcher.map("/f0",                  self.f0)
        self.dispatcher.map("/f0_confidence",       self.f0_confidence)
        self.dispatcher.map("/phonemes",            self.phonemes)
        self.dispatcher.map("/dataset_loudness",    self.set_dataset_loudness)
        self.dispatcher.map("/user_loudness",       self.set_user_loudness)
        self.dispatcher.map("/loudness_extrapolate",self.set_loudness_extrapolate)        
        self.dispatcher.map("/dataset_mfcc-*",      self.set_dataset_mfcc)
        self.dispatcher.map("/user_mfcc-*",         self.set_user_mfcc)
        self.dispatcher.map("/mfcc_extrapolate",    self.set_mfcc_extrapolate)
        self.dispatcher.map("/pitch_scaler",        self.set_pitch_scaler)
        self.dispatcher.map("/loudness_scaler",     self.set_loudness_scaler)


    def start(self):
        print("\nOSC server on {}\n".format(self.server.server_address))
        Thread(target=self.__run_server, args=()).start()

    def __run_server(self):
        # Blocks process forever
        self.server.serve_forever()

    def load_model(self):
        if self.model == None:
            print("No model given")
        else:
            self.model.load(self.voices['voice0']['audio_features'])

    def default_handler(self, address, *args):
        print(f"No call for message: {address}: {args}")

    def interpolate(self, inp, fi):
        # i, f = int(fi // 1), fi % 1  # Split floating-point index into whole & fractional parts.
        # j = i+1 if f > 0 else i  # Avoid index error.
        # return (1-f) * inp[i] + f * inp[j]

        input_list  = inp
        new_len     = fi
        zoom_factor = new_len / len(input_list)

        output_list = interpolation.zoom(input_list, zoom_factor, mode='reflect')

        return output_list

    def predict(self, address, *args):
        if self.model == None:
            print("No model given")
        else:
            voice = args[0]

            if self.voices[voice]['audio_features']['num_samples'] >= 250:

                user_ld_amount = self.voices[voice]['user_loudness'] * (1.0 - self.voices[voice]['loudness_extrapolate'])
                dataset_ld_amount = self.voices[voice]['dataset_loudness'] * self.voices[voice]['loudness_extrapolate']
                self.voices[voice]['audio_features']['loudness_db'] = user_ld_amount + dataset_ld_amount
                # self.voices[voice]['audio_features']['loudness_db'] = (self.voices[voice]['user_loudness'] * (1.0 - self.voices[voice]['loudness_extrapolate'])) + (self.voices[voice]['dataset_loudness'] * self.voices[voice]['loudness_extrapolate'])

                user_mfcc_amount = self.voices[voice]['user_mfcc'] * (1.0 - self.voices[voice]['mfcc_extrapolate'])
                dataset_mfcc_amount = self.voices[voice]['dataset_mfcc'] * self.voices[voice]['mfcc_extrapolate']
                self.voices[voice]['audio_features']['mfccs'] = user_mfcc_amount + dataset_mfcc_amount          
                # self.voices[voice]['audio_features']['mfccs'] = (self.voices[voice]['user_mfcc'] * (1.0 - self.voices[voice]['mfcc_extrapolate'])) + (self.voices[voice]['dataset_mfcc'] * self.voices[voice]['mfcc_extrapolate'])

                # print(self.audio_features['mfccs'].shape)

                exported_filepath = self.model.predict(self.voices[voice]['audio_features'], voice)
                self.client.send_message("/exported_file", [voice, exported_filepath])

                # print(exported_filepath)
                # print(self.voices[voice]['audio_features'])

            else:
                print("\nLength of the provided audio features is too short")

    def num_samples(self, address, *args):
        voice = args[0]
        self.voices[voice]['audio_features']['num_samples'] = args[1]
        print("\nNum Samples", voice, " :", self.voices[voice]['audio_features']['num_samples'])
        # print(type(self.audio_features['num_samples']))

    def f0(self, address, *args):
        voice       = args[0]
        filepath    = args[1]
        input       = []

        with open(filepath) as file:
            for line in file:
                input.append(float(line))

        new_len = int(math.ceil(self.voices[voice]['audio_features']['num_samples'] / 64))
        resized = self.interpolate(input, new_len)
        resized = np.clip(resized, 0, 16000)
        formated = np.asarray(resized, dtype=float)
        self.voices[voice]['audio_features']['f0_hz'] = formated * self.voices[voice]['pitch_scaler']
        print("F0 Length", voice, " :", len(input), "->", len(self.voices[voice]['audio_features']['f0_hz']))

        # print(self.voices[voice]['audio_features']['f0_hz'])

        # new_len = int(math.ceil(self.voices[voice]['audio_features']['num_samples'] / 64))
        # delta = (len(input)-1) / (new_len-1)
        # resized = [self.interpolate(input, i*delta) for i in range(new_len)]
        # formated = np.asarray(resized, dtype=float)
        # self.voices[voice]['audio_features']['f0_hz'] = formated * self.voices[voice]['pitch_scaler']
        # print("F0 Length", voice, " :", len(input), "->", len(self.voices[voice]['audio_features']['f0_hz']))
        # # print(self.audio_features['f0_hz'].shape)

        self.voices[voice]['audio_features']['f0_confidence'] = np.clip(self.voices[voice]['audio_features']['f0_hz'], 0 , 1)


    def f0_confidence(self, address, *args):
        print('Using f0_hz clipped between 0 and 1 for f0_confidence')
        # input = list(args)
        # new_len = int(self.audio_features['num_samples'] / 64)
        # delta = (len(input)-1) / (new_len-1)
        # resized = [self.interpolate(input, i*delta) for i in range(new_len)]
        # formated = np.asarray(resized, dtype=float)
        # self.audio_features['f0_confidence'] = formated
        # print("F0_confidence Length :", len(input), "->", len(self.audio_features['f0_confidence']))
        # # print(self.audio_features['f0_confidence'].shape)

    def phonemes(self, address, *args):
        voice       = args[0]
        filepath    = args[1]
        input       = []

        with open(filepath) as file:
            for line in file:
                input.append(float(line))

        new_len = int(math.ceil(self.voices[voice]['audio_features']['num_samples'] / 64))
        resized = self.interpolate(input, new_len)
        formated = np.asarray(resized)
        self.voices[voice]['audio_features']['phoneme'] = formated
        print("Phoneme Length", voice, " :", len(input), "->", len(self.voices[voice]['audio_features']['phoneme']))
        # print(self.audio_features['phoneme'].shape)

    def set_user_loudness(self, address, *args):
        voice       = args[0]
        filepath    = args[1]
        input       = []

        with open(filepath) as file:
            for line in file:
                input.append(float(line))

        new_len = int(math.ceil(self.voices[voice]['audio_features']['num_samples'] / 64))
        resized = self.interpolate(input, new_len)
        formated = np.asarray(resized, dtype=float)
        self.voices[voice]['user_loudness'] = formated + self.voices[voice]['loudness_scaler']
        print("User Loudness Length", voice, " :", len(input), "->", len(self.voices[voice]['user_loudness']))
        # print(self.audio_features['loudness_db'].shape)

        # print(self.voices[voice]['user_loudness'])

    def set_dataset_loudness(self, address, *args):
        voice       = args[0]
        filepath    = args[1]
        input       = []

        with open(filepath) as file:
            for line in file:
                input.append(float(line))

        new_len = int(math.ceil(self.voices[voice]['audio_features']['num_samples'] / 64))
        resized = self.interpolate(input, new_len)
        formated = np.asarray(resized, dtype=float)
        self.voices[voice]['dataset_loudness'] = formated + self.voices[voice]['loudness_scaler']
        print("Dataset Loudness Length", voice, " :", len(input), "->", len(self.voices[voice]['dataset_loudness']))
        # print(self.audio_features['loudness_db'].shape)

    def set_user_mfcc(self, address, *args):
        voice = args[0]
        filepath    = args[1]
        input       = []

        if address == '/user_mfcc-0':
            self.voices[voice]['user_mfcc_list'] = []

        with open(filepath) as file:
            for line in file:
                input.append(float(line))

        new_len = int(math.ceil(self.voices[voice]['audio_features']['num_samples'] / 500))
        resized = self.interpolate(input, new_len)

        self.voices[voice]['user_mfcc_list'].append(resized)

        if address == '/user_mfcc-29':

            mfcc_list_resized = np.asarray(self.voices[voice]['user_mfcc_list'], dtype=float).T
            mfcc_list_resized = np.expand_dims(mfcc_list_resized, axis=0)
            mfccs = convert_to_tensor(mfcc_list_resized, np.float32)

            self.voices[voice]['audio_features']['audio'] = None
            self.voices[voice]['user_mfcc'] = mfccs

            print("User MFCC Length", voice, " :", len(input), "->", len(self.voices[voice]['user_mfcc'][0]))
        #     # print(mfccs)
        #     # print(self.audio_features['mfccs'][0].shape)

    def set_dataset_mfcc(self, address, *args):
        voice = args[0]
        filepath    = args[1]
        input       = []

        if address == '/dataset_mfcc-0':
            self.voices[voice]['dataset_mfcc_list'] = []

        with open(filepath) as file:
            for line in file:
                input.append(float(line))

        new_len = int(math.ceil(self.voices[voice]['audio_features']['num_samples'] / 500))
        resized = self.interpolate(input, new_len)

        self.voices[voice]['dataset_mfcc_list'].append(resized)

        if address == '/dataset_mfcc-29':

            mfcc_list_resized = np.asarray(self.voices[voice]['dataset_mfcc_list'], dtype=float).T
            mfcc_list_resized = np.expand_dims(mfcc_list_resized, axis=0)
            mfccs = convert_to_tensor(mfcc_list_resized, np.float32)

            self.voices[voice]['audio_features']['audio'] = None
            self.voices[voice]['dataset_mfcc'] = mfccs

            print("Dataset MFCC Length", voice, " :", len(input), "->", len(self.voices[voice]['dataset_mfcc'][0]))
            # print(mfccs)
            # print(self.audio_features['mfccs'][0].shape)

    def set_loudness_extrapolate(self, address, *args):
        voice = args[0]
        self.voices[voice]['loudness_extrapolate'] = args[1]
        print("Loudness Extrapolate", voice, " :", self.voices[voice]['loudness_extrapolate'])
        # print(type(self.audio_features['num_samples']))

    def set_mfcc_extrapolate(self, address, *args):
        voice = args[0]
        self.voices[voice]['mfcc_extrapolate'] = args[1]
        print("MFCC Extrapolate", voice, " :", self.voices[voice]['mfcc_extrapolate'])
        # print(type(self.audio_features['num_samples']))

    def set_pitch_scaler(self, address, *args):
        voice = args[0]
        self.voices[voice]['pitch_scaler'] = args[1]
        print("Pitch scaler", voice, " :", self.voices[voice]['pitch_scaler'])

    def set_loudness_scaler(self, address, *args):
        voice = args[0]
        self.voices[voice]['loudness_scaler'] = args[1]
        print("Loudness scaler", voice, " :", self.voices[voice]['loudness_scaler'])