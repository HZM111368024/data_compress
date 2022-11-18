import cv2 as cv
import matplotlib.pyplot as plt
import skimage.measure
import pandas as pd
import numpy as np


src = cv.imread("./Lennagrey.bmp", 0)
cv.imshow("src", src)
plt.hist(src.ravel(), 256, density=True)
print(skimage.measure.shannon_entropy(src))
plt.show()

print(src)