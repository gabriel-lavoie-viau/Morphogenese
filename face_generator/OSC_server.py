from threading import Thread
import time

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


class OSC_server:

    def __init__(self, face_interpolator, ip="127.0.0.1", port=9001):
        # Reference to the face_interpolation class instantiation
        self.face_interpolator  = face_interpolator

        # Main script params
        self.blur_amount = 0.75
        self.mask_amount = 0.75
        self.led_ring_luminosity = 0.0
        self.state = 0
        self.eyes_are_open = False

        # OSC paramters 
        self.ip     = ip
        self.port   = port
        self.client = SimpleUDPClient("127.0.0.1", 9000)

        # Create dispatcher
        self.dispatcher = Dispatcher()
        # Create server
        self.server = BlockingOSCUDPServer((self.ip, self.port), self.dispatcher)

        # Create function for unknown messages
        self.dispatcher.set_default_handler(self.default_handler)

        # Calls to functions
        self.dispatcher.map("/goto_rdm_user_face",          self.goto_rdm_user_face)
        self.dispatcher.map("/goto_rdm_dataset_face",       self.goto_rdm_dataset_face)
        self.dispatcher.map("/goto_user_face",              self.goto_user_face)
        self.dispatcher.map("/goto_dataset_face",           self.goto_dataset_face)
        self.dispatcher.map("/generate_rdm_dataset_face",   self.generate_rdm_dataset_face)
        self.dispatcher.map("/user_vs_dataset",             self.user_vs_dataset)
        self.dispatcher.map("/jitter_amplitude",            self.jitter_amp)
        self.dispatcher.map("/jitter_speed",                self.jitter_speed)
        self.dispatcher.map("/directions",                  self.directions)
        self.dispatcher.map("/save_face",                   self.save_face)
        self.dispatcher.map("/save_directions",             self.save_directions)
        self.dispatcher.map("/blur_amount",                 self.set_blur_amount)
        self.dispatcher.map("/mask_amount",                 self.set_mask_amount)
        self.dispatcher.map("/led_ring_luminosity",         self.set_led_ring_luminosity)
        self.dispatcher.map("/state",                       self.set_state)
        self.dispatcher.map("/export_image",                self.export_image)


    def start(self):
        print("OSC server on {}".format(self.server.server_address))
        Thread(target=self.__run_server, args=()).start()

    def __run_server(self):
        # Blocks process forever
        self.server.serve_forever()

    def osc_send(self, address, *args):
        message = round(args[0], 2)
        self.client.send_message(address, message)

    def default_handler(self, address, *args):
        print(f"No call for message: {address}: {args}")

    def goto_rdm_user_face(self, address, *args):
        self.face_interpolator.change_user_face()

    def goto_rdm_dataset_face(self, address, *args):
        self.face_interpolator.change_dataset_face()

    def goto_user_face(self, address, *args):
        self.face_interpolator.change_user_face(args[0])

    def goto_dataset_face(self, address, *args):
        self.face_interpolator.change_dataset_face(args[0])

    def generate_rdm_dataset_face(self, address, *args):
        self.face_interpolator.change_dataset_face(generate_random=True)

    def user_vs_dataset(self, address, *args):
        self.face_interpolator.set_user_vs_dataset_face_interpolation(args[0])

    def jitter_amp(self, address, *args):
        self.face_interpolator.set_jitter_amplitude(args[0])

    def jitter_speed(self, address, *args):
        self.face_interpolator.set_jitter_speed(args[0])

    def directions(self, address, *args):
        self.face_interpolator.set_directions(args[0], args[1])

        if args[0] == 'eye_openness':
            if args[1] < 0.1:
                self.eyes_are_open = True
            if args[1] > 35.0:
                self.eyes_are_open = False

    def save_face(self, address, *args):
        if args[0] > 0.5:
            self.face_interpolator.encode_face()
        else:
            self.face_interpolator.save_latents(is_user_face=False)

    def save_directions(self, address, *args):
        self.face_interpolator.save_directions(args[0])

    def set_blur_amount(self, address, *args):
        self.blur_amount = args[0]

    def set_mask_amount(self, address, *args):
        self.mask_amount = args[0]

    def set_led_ring_luminosity(self, address, *args):
        self.led_ring_luminosity = args[0]

    def set_state(self, address, *args):
        self.state = int(args[0])

    def export_image(self, address, *args):
        self.face_interpolator.export_image()