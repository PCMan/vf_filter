#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
from ptsa.ptsa import emd

def main():
    x = np.linspace(0, np.pi * 30, 1000)
    y1 = np.sin(x)
    y2 = np.sin(x / 2)
    y3 = np.sin(x / 4)
    y = y1 + y2 + y3
    f, ax = plt.subplots(4, 1, sharex=True, sharey=True)
    ax[0].plot(x, y, color="k")
    ax[0].set_title("Original Signal", fontsize=22)
    
    ue = emd._get_upper_spline(y)
    le = -emd._get_upper_spline(-y)
    ax[1].plot(x, y, color="k")
    ax[1].plot(x, ue, color="r", linestyle="-")
    ax[1].plot(x, le, color="g", linestyle="-")
    avg = (ue + le) / 2
    ax[1].plot(x, avg, color="b", linestyle="-")
    ax[1].set_title("Calculate Mean of Upper and Lower Envolopes", fontsize=22)
    
    imf1 = y - avg
    ax[2].plot(x, imf1, color="k")
    ax[2].set_title("Intrinsic Mode Function (IMF) = Original Signal - Mean", fontsize=22)

    res = y - imf1
    ax[3].plot(x, res, color="k")
    ax[3].set_title("Residual = Original Signal - IMF", fontsize=22)

    plt.show()


if __name__ == "__main__":
    main()
