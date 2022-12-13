from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

import time

ip = "127.0.0.1"
## MaxMSP port number
port_client = 1337

## Create client
client = SimpleUDPClient(ip, port_client)

i = 0
while i < 10:
    print(i)
    client.send_message("/from_python_sender", i)
    i += 1
    time.sleep(2)