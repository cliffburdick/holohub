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
from matplotlib.widgets import Slider
import struct
import subprocess


L = 1228800
BW = 100
FFT_SIZE = 16384

np.set_printoptions(threshold=np.inf)


rxq_psd = Queue()
txq = Queue()
nats_inst = nats_async("10.110.102.191", {'spec_output':rxq_psd}, txq)
t = threading.Thread(target=nats_inst.start_async_loop, daemon=True)
t.start()    
txq.put_nowait(subscribe(subject="spec_output"))

Fs = 2949120000/3
l1 = FFT_SIZE
x = np.linspace(-Fs/2, Fs/2 - Fs/l1, l1)/1e6
#fig = plt.figure()
fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.75)

axfreq = fig.add_axes([0.85, 0.15, 0.0225, 0.70])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [kHz]',
    valmin=0,
    valmax=10e3,
    valinit=0,
    orientation="vertical"
)

def update_slider(val):
    #line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    #fig.canvas.draw_idle()
    print(f'Updating frequency to {freq_slider.val}')
    subprocess.run(["python3", "adi_mxfe.py", "-a", "192.168.0.2:8192", "-f", str(freq_slider.val)]) 

freq_slider.on_changed(update_slider)

line_data = np.random.randn(FFT_SIZE)
line, = ax1.plot(x, line_data)
plt.grid()

def init():
  ax1.set_ylim(-100, 80)
  ax1.set_title('JESD output of MxFE')
  ax1.set_ylabel('Power (dB)')
  ax1.set_xlabel('Frequency (MHz)')
  line.set_ydata(np.ma.array(np.random.randn(FFT_SIZE), mask=True))
  return line,

def animate(i):
  try:
    obj = rxq_psd.get(block=False)
    data_len = FFT_SIZE * 4
    #mtype, data = struct.unpack(f'@I{data_len}s', obj)
    (data,) = struct.unpack(f'@{data_len}s', obj)
    y = np.frombuffer(data, dtype=np.float32)
    #print(y)
    #ax1.clear()
    line_data = y
    line.set_ydata(line_data)

    print("Update")
  except Empty:
    pass

  return line,
    


ani = animation.FuncAnimation(fig, animate, interval=50, blit=True, init_func=init)
plt.show()
