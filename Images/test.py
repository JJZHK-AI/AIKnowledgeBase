import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
image = cv2.imread("Images/01/09/02.jpg", 1)

def affine(_img, parameter):
    a,b,c,d,tx,ty = parameter
    H,W,C = _img.shape
    img = np.zeros((H+2, W+2, C), dtype=np.float32)
    img[1:H+1, 1:W+1] = _img

	# get new image shape
    H_new = np.round(H * d).astype(np.int)
    W_new = np.round(W * a).astype(np.int)
    out = np.zeros((H_new+1, W_new+1, C), dtype=np.float32)

	# get position of new image
    x_new = np.tile(np.arange(W_new), (H_new, 1))
    y_new = np.arange(H_new).repeat(W_new).reshape(H_new, -1)

	# get position of original image by affine
    adbc = a * d - b * c
    x = np.round((d * x_new  - b * y_new) / adbc).astype(np.int) - tx + 1
    y = np.round((-c * x_new + a * y_new) / adbc).astype(np.int) - ty + 1
    
    x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)

	# assgin pixcel to new image
    out[y_new, x_new] = img[y, x]
    out = out[:H_new, :W_new]
    out = out.astype(np.uint8)
    return out

def myath(img):
    H,W,C = img.shapes
    paramters = np.array([[1,0,30],[0,1,30],[0,0,1]], dtype=np.unit8)
    

plt.figure(figsize=(10,10), facecolor='w')

plt.subplot(3, 3, 1)
plt.title(u"原始图像")
show_img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
plt.imshow(show_img1)

plt.subplot(3, 3, 2)
plt.title(u"平移")
show_img2 = cv2.cvtColor(affine(image, parameter=(1,0,0,1,30,-30)), cv2.COLOR_BGR2RGB) 
plt.imshow(show_img2)

plt.subplot(3, 3, 3)
plt.title(u"关于原点放大")
show_img2 = cv2.cvtColor(affine(image, parameter=(1.5,0,0,0.5,0,0)), cv2.COLOR_BGR2RGB) 
plt.imshow(show_img2)

plt.subplot(3, 3, 4)
plt.title(u"关于原点旋转")
show_img2 = cv2.cvtColor(affine(image, parameter=(math.cos(math.pi / 4),math.sin(math.pi / 4),-math.sin(math.pi / 4),math.cos(math.pi / 4),0,0)), cv2.COLOR_BGR2RGB) 
plt.imshow(show_img2)

plt.subplot(3, 3, 5)
plt.title(u"斜向拉伸-x轴")
show_img2 = cv2.cvtColor(affine(image, parameter=(1,math.tan(math.pi / 4),0,1,0,0)), cv2.COLOR_BGR2RGB) 
plt.imshow(show_img2)

plt.subplot(3, 3, 6)
plt.title(u"斜向拉伸-y轴")
show_img2 = cv2.cvtColor(affine(image, parameter=(1,0,math.tan(math.pi / 4),1,0,0)), cv2.COLOR_BGR2RGB) 
plt.imshow(show_img2)

# ax7 = plt.subplot(3, 3, 7)
# ax7.set_xlim((-400, 400))
# ax7.set_ylim((-600, 600))
# plt.title(u"关于原点翻转")
# show_img2 = cv2.cvtColor(affine(image, parameter=(-1,0,0,-1,0,0)), cv2.COLOR_BGR2RGB) 
# plt.imshow(show_img2)

# plt.subplot(3, 3, 8)
# plt.title(u"关于x轴翻转")
# show_img2 = cv2.cvtColor(affine(image, parameter=(1,0,0,-1,0,0)), cv2.COLOR_BGR2RGB) 
# plt.imshow(show_img2)

# plt.subplot(3, 3, 9)
# plt.title(u"关于y轴翻转")
# show_img2 = cv2.cvtColor(affine(image, parameter=(-1,0,0,1,0,0)), cv2.COLOR_BGR2RGB) 
# plt.imshow(show_img2)
plt.show()