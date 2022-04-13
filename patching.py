import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans
import time

def patch_img(img, k = 50):
    img = np.asarray(img[0])
    img = np.moveaxis(img,0,-1)

    start = time.time()
    X = []
    for x,row in enumerate(img):
        for y,col in enumerate(row):
            X.append([x/img.shape[0],y/img.shape[1],col[0], col[1], col[2]])
    print("imgprep",time.time()-start)
    start = time.time()
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit_predict(X)
    print("kmeans",time.time()-start)
    start = time.time()
    new_img = np.zeros(img.shape[:2])
    for x,row in enumerate(img):
        for y,col in enumerate(row):
            new_img[x,y] = kmeans[y + x* len(row)]
    print("imgfing",time.time()-start)
    start = time.time()
    return new_img
#     new_img = new_img *4
#     plt.imshow(new_img)
#     plt.show()
    