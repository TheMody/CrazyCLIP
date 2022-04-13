import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def process_mask(mask):
#     mask = mask *255
#     mask = mask.astype(int)
#     num, bin = np.histogram(mask, bins = 256)
#     print(len(num),len(bin))
#     plt.bar(bin[:256],num)
#     plt.show()
    mask = mask > 0.75
#     print(mask)
#     plt.imshow(mask)
#     plt.show()

    kernel = np.ones((3,3),np.uint8)
    dilatedmask = cv.dilate(mask.astype(np.uint8),kernel,iterations = 2)
    erosion = cv.erode(dilatedmask,kernel,iterations = 4)
    dilatedmask = cv.dilate(erosion,kernel,iterations = 2)
    
    plt.imshow(dilatedmask)
    plt.show()
    return
    