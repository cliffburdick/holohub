import struct
import numpy as np
import numpy as cp
from queue import Queue, Empty
import threading
from adi_nats import nats_async
from common_msg import subscribe, external_message
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import struct
import sys
import time


L = 1228800
BW = 100
FFT_SIZE = 1024

np.set_printoptions(threshold=np.inf)


rxq_psd = Queue()
txq = Queue()
nats_inst = nats_async("10.110.102.191", {'spec_output':rxq_psd}, txq)
t = threading.Thread(target=nats_inst.start_async_loop, daemon=True)
t.start()    
txq.put_nowait(subscribe(subject="spec_output"))

Fs = 2949120000/3
l1 = 1024
x = np.linspace(-Fs/2, Fs/2 - Fs/l1, l1)/1e6
#fig = plt.figure()
fig, ax1 = plt.subplots()
line, = ax1.plot(x, np.random.randn(FFT_SIZE))
plt.grid()

def init():
  ax1.set_ylim(-120, 0)
  ax1.set_title('JESD output of MxFE')
  ax1.set_ylabel('Power (dB)')
  ax1.set_xlabel('Frequency (MHz)')
  line.set_ydata(np.ma.array(np.random.randn(FFT_SIZE), mask=True))
  return line,

def animate(i):
  try:
    obj = rxq_psd.get(block=False)
    data_len = 1024 * 4
    #mtype, data = struct.unpack(f'@I{data_len}s', obj)
    (data,) = struct.unpack(f'@{data_len}s', obj)
    y = np.frombuffer(data, dtype=np.float32)
    #ax1.clear()

    line.set_ydata(y)
  except Empty:
      print("here")

  return line,
    


ani = animation.FuncAnimation(fig, animate, interval=50, blit=True, init_func=init)
plt.show()
