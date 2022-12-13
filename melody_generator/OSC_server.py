from threading import Thread
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# import model

class OSC_server:

    def __init__(self, ip="127.0.0.1", port=8001, model=None):
        # OSC paramters 
        self.ip     = ip
        self.port   = port
        self.client = SimpleUDPClient("127.0.0.1", 8000)

        self.model = model

        # Prediction parameters
        num_voices      = 6
        self.voices     = {}

        for i in range(num_voices):
            voice_name = 'voice' + str(i)
            self.voices[voice_name] = {}

        for voice in self.voices:
            self.voices[voice]['output_dir']            = '../melody_generator/generated' # Path relative to the pure data main patch
            self.voices[voice]['num_outputs']           = 1
            self.voices[voice]['num_steps']             = 100
            self.voices[voice]['temperature']           = 1.0
            self.voices[voice]['primer_midi']           = './generated/voice_primer.mid' #Path relative to the melody generator script
            self.voices[voice]['primer_steps_scaler']   = 1.0
            self.voices[voice]['num_steps_scaler']      = 1.0
            self.voices[voice]['primer_toggle']         = True

        # Create dispatcher
        self.dispatcher = Dispatcher()
        # Create server
        self.server = BlockingOSCUDPServer((self.ip, self.port), self.dispatcher)
        # Create function for unknown messages
        self.dispatcher.set_default_handler(self.default_handler)

        # Loading model at init
        if self.model == None:
            print("\nNo model given")
        else:
            self.model.load()

        # Calls to functions
        self.dispatcher.map("/predict",             self.predict)
        self.dispatcher.map("/output_dir",          self.set_output_dir)
        self.dispatcher.map("/num_outputs",         self.set_num_outputs)
        self.dispatcher.map("/num_steps",           self.set_num_steps)
        self.dispatcher.map("/temperature",         self.set_temperature)
        self.dispatcher.map("/primer_midi",         self.set_primer_midi)
        self.dispatcher.map("/primer_steps_scaler", self.set_primer_steps_scaler)
        self.dispatcher.map("/num_steps_scaler",    self.set_num_steps_scaler)
        self.dispatcher.map("/primer_toggle",       self.set_primer_toggle)


    def start(self):
        print("\nOSC server on {}".format(self.server.server_address))
        Thread(target=self.__run_server, args=()).start()

    def __run_server(self):
        # Blocks process forever
        self.server.serve_forever()

    def default_handler(self, address, *args):
        print(f"\nNo call for message: {address}: {args}")

    def predict(self, address, *args):
        voice = args[0]

        midi_filepath = self.model.predict( output_dir          = self.voices[voice]['output_dir'], 
                                            num_outputs         = self.voices[voice]['num_outputs'], 
                                            num_steps           = self.voices[voice]['num_steps'], 
                                            temperature         = self.voices[voice]['temperature'], 
                                            primer_midi         = self.voices[voice]['primer_midi'],
                                            primer_steps_scaler = self.voices[voice]['primer_steps_scaler'], 
                                            num_steps_scaler    = self.voices[voice]['num_steps_scaler'],
                                            primer_toggle       = self.voices[voice]['primer_toggle'],
                                            voice_name          = voice)

        self.client.send_message("/exported_file", [voice, midi_filepath])
        print('\nMidi file created and saved at: ', midi_filepath)

    def set_output_dir(self, address, *args):
        self.voices[args[0]]['output_dir'] = args[1]
        print("Output directory :", args[0], ":", self.voices[args[0]]['output_dir'])

    def set_num_outputs(self, address, *args):
        self.voices[args[0]]['num_outputs'] = args[1]
        print("Number of outputs:", args[0], ":", self.voices[args[0]]['num_outputs'])

    def set_num_steps(self, address, *args):
        self.voices[args[0]]['num_steps'] = args[1]
        print("Number of steps :", args[0], ":", self.voices[args[0]]['num_steps'])

    def set_temperature(self, address, *args):
        self.voices[args[0]]['temperature'] = args[1]
        print("Temperature :", args[0], ":", self.voices[args[0]]['temperature'])

    def set_primer_midi(self, address, *args):
        if args[1] == 'none':
            self.voices[args[0]]['primer_midi'] = ''
            print("Primer midi file :", args[0], ":", "None")
        else:
            self.voices[args[0]]['primer_midi'] = args[1]
            print("Primer midi file :", args[0], ":", self.voices[args[0]]['primer_midi'])

    def set_primer_steps_scaler(self, address, *args):
        self.voices[args[0]]['primer_steps_scaler'] = args[1]
        print("Primer steps scaler :", args[0], ":", self.voices[args[0]]['primer_steps_scaler'])

    def set_num_steps_scaler(self, address, *args):
        self.voices[args[0]]['num_steps_scaler'] = args[1]
        print("Num steps scaler :", args[0], ":", self.voices[args[0]]['num_steps_scaler'])

    def set_primer_toggle(self, address, *args):
        if (args[1] > 0.5):
            self.voices[args[0]]['primer_toggle'] = True
        else:
            self.voices[args[0]]['primer_toggle'] = False
        print("Primer toggle", args[0], ":", self.voices[args[0]]['primer_toggle'])