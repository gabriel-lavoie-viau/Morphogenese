from model_ddsp import *
from OSC_server import *

voices_ddsp = model_ddsp()
osc = OSC_server(model=voices_ddsp)

osc.start()
osc.load_model()