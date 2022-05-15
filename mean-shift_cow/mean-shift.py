import time
import os
import random
import math
import torch
import numpy as np


# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    return torch.sqrt(((X-x) **2).sum(1))

def distance_batch(x, X):
    x = x.unsqueeze(1).repeat(1, X.shape[0],1)
    return torch.sqrt(((X-x) **2).sum(2))

def gaussian(dist, bandwidth):
    return torch.exp(-(dist**2/(2*(bandwidth**2))))

def update_point(weight, X):
    return torch.sum(X * weight.unsqueeze(1).repeat(1, X.shape[1]), dim=0)/weight.sum(0)

def update_point_batch(weight, X):
    weighted_sum = (X.unsqueeze(0).repeat(weight.shape[0],1,1) * weight.unsqueeze(2).repeat(1,1, X.shape[1])).sum(1)
    return torch.t(torch.div(torch.t(weighted_sum),weight.sum(1)))

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    batch_size = 50
    for i in range(0,len(X),batch_size):
        x = X[i:i+batch_size]
        dist = distance_batch(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i:i+batch_size] = update_point_batch(weight, X)
    return X_

def meanshift(X):
    X = X.clone()
    for i in range(20):
        X = meanshift_step(X)   # slow implementation
        #X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
#X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
X = torch.from_numpy(image_lab).cuda()
X = meanshift(X).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)


#With cuda (from colab)
# slow mode: Elapsed time for mean-shift: 21.242610216140747
# fast mode: Elapsed time for mean-shift: 0.9464685916900635
