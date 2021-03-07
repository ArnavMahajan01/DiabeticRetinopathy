import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ksize = 20
sigma = 10
theta = 0
lamda = 2
gamma = 0.5
phi = 0


img = cv2.imread("./preprocessed_dataset/training/21.tif", flags=cv2.IMREAD_GRAYSCALE)
img2 = img.reshape(-1)
df = pd.DataFrame()
df['original image'] = img2

num = 1
kernels = []
for theta in range(4):
    theta = theta/4 * np.pi
    for sigma in (1, 3, 5):
        for lamda in np.arange(0, np.pi, np.pi/4):
            for gamma in (0.05, 0.5):
                gabor_label = 'Gabor' + str(num) 
                ksize = 15
                phi = 0.8
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                kernels.append(kernel)

                fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)

                cv2.imwrite('./gabor/training/22/'+gabor_label+'.tif', filtered_img.reshape(img.shape))

                df[gabor_label] = filtered_img
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma, ': phi=', phi)
                num += 1

print(df.head())

# kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_64F)
# fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

cv2.waitKey(0)
cv2.destroyAllWindows()

