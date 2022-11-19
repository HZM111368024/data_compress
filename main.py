import cv2 as cv
import matplotlib.pyplot as plt
import skimage.measure
import pandas as pd
import numpy as np
import math

# use opencv calculate entropy
src = cv.imread("./Lennagrey.bmp", 0)
cv.imshow("src", src)
plt.hist(src.ravel(), 256, density=True)


# Independent calculation entropy
def entropy(labels):
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0
    # Compute entropy
    for i in probs:
        ent -= i * math.log(i, 2)
    return ent


ori_image = []
construct_image = []

for i in range(0, 512):
    for j in range(0, 512):
        ori_image.append(src[i][j])

# construct a difference image of Lennagrey.bmp
for i in range(0, 512):
    for j in range(0, 512):
        if i == 0 and j == 0:
            diff = 0
        elif j == 0:
            diff = int(src[i][j]) - int(src[i - 1][j])
        else:
            diff = int(src[i][j]) - int(src[i][j - 1])
        construct_image.append(diff)

# show entropy by api read image
print("Q3 Entropy by api : " + str(skimage.measure.shannon_entropy(src)))
# calculate entropy by function
print("Q3 Entropy by log : " + str(entropy(ori_image)))
# calculate construct image by function
print("Q4 Entropy by log : " + str(entropy(construct_image)))
# show Probability
plt.show()
