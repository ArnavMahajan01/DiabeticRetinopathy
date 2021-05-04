import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

c = 1
for theta in range(16):
        theta = theta/16 * np.pi
        for lamda in np.arange(1, np.pi+1, np.pi/4):
            ksize = 5
            phi = 0.8
            kernel = cv2.getGaborKernel((ksize, ksize), 3, theta, lamda, 0.05, phi, ktype=cv2.CV_64F)
            kernel_resized = cv2.resize(kernel, (400, 400))
            #cv2.imshow(str(c), kernel_resized)
            plt.imsave("gabor"+str(c)+".png", kernel_resized)
            c = c + 1