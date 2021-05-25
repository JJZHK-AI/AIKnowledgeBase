import cv2
import matplotlib.pyplot as plt
import numpy as np

import torchvision.models as models
import torch
from JLib.pyn.layers import Conv2D
from JLib.pyn.functions import preprocess_image

plt.figure(figsize=(10,10), facecolor='w')
imagePath = "data/ConvVisible01.jpg"

img = cv2.imread(imagePath)
basicImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
originImg = cv2.resize(basicImage, (224, 224))
plt.subplot(1, 2, 1)
plt.title('Origin')
plt.imshow(originImg)

convImage = np.expand_dims(originImg, axis=0)
conv = Conv2D(convImage.shape, 12, 3, 1, 'SAME')
convImage = conv.forward(convImage) # 这里使用的是初始化权重，还没有进行过迭代
convImage = convImage[:, :, :, 0]
print(convImage.shape)
feature = np.reshape(convImage, newshape=(convImage.shape[1], convImage.shape[2]))
feature = 1.0 / (1 + np.exp(-1 * feature))
feature = np.round(feature * 255)
plt.subplot(1, 2, 2)
plt.title('Conv2D')
plt.imshow(feature, cmap='gray')
plt.show()