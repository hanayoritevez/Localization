
import numpy as np
from ctypes.wintypes import HBITMAP
from curses.ascii import STX
from imp import C_EXTENSION
from locale import DAY_2, DAY_3
from re import A
from signal import valid_signals
from tkinter import E
from unittest.main import MAIN_EXAMPLES
from zlib import Z_BLOCK

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from pynput import keyboard

import bisect

import sensor

def move():


    # visualize




   
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    while True:

        # vx,vth =  keyPressed()
       # Collect events until released
        with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
            listener.join()






        # plt.show()
  



def on_press(key):
    try:
        print('Alphanumeric key pressed: {0} '.format(
            key.char))
    except AttributeError:
        print('special key pressed: {0}'.format(
            key))

def on_release(key):
    print('Key released: {0}'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False



if __name__ == '__main__':
    move()