from melody_generator import *
from OSC_server import *

bundle_path = './bundles/attention_rnn.mag'
melody_maker = melody_generator(bundle_file=bundle_path)

osc = OSC_server(model=melody_maker)
osc.start()