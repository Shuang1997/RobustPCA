import os

path = "/home/guoshuang/Project/OptimizationCourse/experiment/Robust_PCA/imgs"
dirs = os.listdir(path)
dirs.sort()
file_handle=open('imgs.txt',mode='w')
for file in dirs:
    file_handle.write(file + '\n')
