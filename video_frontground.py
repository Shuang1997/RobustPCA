import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from PIL import Image

from RobustPCA import RobustPCA

def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])

f = open("imgs.txt")

img_dir_path = "/home/guoshuang/Project/OptimizationCourse/experiment/Robust_PCA/imgs/"


n_frame = 180
height = 36
width = 64


X = np.zeros([n_frame, height * width])
#print(X.shape)

for i in range(0, n_frame):
    line = f.readline()
    line = line.strip('\n')
    #print(line)

    img = mpimg.imread(img_dir_path + line)

    img = rgb2gray(img)

    X[i] = img.reshape(1, -1, order="F")


RobustPCA = RobustPCA(X)
L, S = RobustPCA.iterate()

for i in range(0, n_frame):
    plt.subplot(131)
    plt.title("original image")
    frame1 = X[i].reshape(height, -1, order="F")
    plt.imshow(frame1)

    plt.subplot(132)
    plt.title("background")
    frame2 = L[i].reshape(height, -1, order="F")
    plt.imshow(frame2)

    plt.subplot(133)
    plt.title("front")
    frame3 = np.abs(S[i]).reshape(height, -1, order="F")
    plt.imshow(frame3)

    #plt.imsave("video_frontbackground.png")
    plt.show()

    #plt.imshow(frame)
    #plt.show()