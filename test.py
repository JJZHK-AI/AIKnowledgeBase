from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
# from dask_ml.cluster import KMeans
from sklearn.cluster import KMeans
def vector_of_pixels(np_im):
    pixels = []
    for row in np_im:
        for pixel in row:
            pixels.append(pixel)
    return np.asarray(pixels)

def pixels_from_path(file_path):
    im = Image.open(file_path)
    np_im = np.array(im)
    return np_im

def clustered_pixels(x_fit, pixels):
    labels = x_fit.predict(pixels)
    res= x_fit.cluster_centers_[labels] 
    return res

def reshape_pixels(pixels, width, height):
    resulting_pixels = [[] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            resulting_pixels[i].append(pixels[width*i+j])
    return np.asarray(resulting_pixels)

def compress_image(np_im, k):
    
    x,y = np_im.shape[0],np_im.shape[1]
    pixels = vector_of_pixels(np_im)
    
    km = KMeans(n_clusters=k, init='k-means++')
    x_fit = km.fit(pixels)

    clust_pixels = clustered_pixels(x_fit, pixels)

    x_floor = reshape_pixels(clust_pixels, y, x)
    return x_floor

file_path = "Images/01/09/02.jpg"
np_im = pixels_from_path(file_path)

plt.figure(figsize=(10,5), facecolor='w')
plt.subplot(2, 2, 1) 
plt.imshow(np_im)

plt.subplot(2, 2, 2)
k2Image = compress_image(np_im, 2)
plt.imshow(k2Image / 255)

plt.subplot(2, 2, 3)
k5Image = compress_image(np_im, 5)
plt.imshow(k5Image / 255)

plt.subplot(2, 2, 4)
k10Image = compress_image(np_im, 10)
plt.imshow(k10Image / 255)
plt.show()