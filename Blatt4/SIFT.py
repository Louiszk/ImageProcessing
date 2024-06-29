import cv2
import matplotlib.pyplot as plt
from SIFT_algo import SIFT_Algorithm
import own_sift as ownSIFT
from SIFT_Params import SIFT_Params
import SIFT_Visualization as vis

image = cv2.imread('image.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255

sift_params = SIFT_Params() 

scale_space, deltas, sigmas = ownSIFT.create_scale_space(gray, sift_params)

dogs = ownSIFT.create_dogs(scale_space, sift_params)

extremas = ownSIFT.find_discrete_extremas(dogs, sift_params, sigmas, deltas)

finetuned_extremas = SIFT_Algorithm.taylor_expansion(extremas, dogs, sift_params, deltas, sigmas)

filtered_extremas = SIFT_Algorithm.filter_extremas(finetuned_extremas, dogs, sift_params)

gradients = SIFT_Algorithm.gradient_2d(scale_space, sift_params)

keypoints_with_orientation = SIFT_Algorithm.assign_orientations(filtered_extremas, scale_space, sift_params, gradients, deltas)

vis.visualize_keypoints(scale_space, keypoints_with_orientation, deltas, "Keypoints", True, True, True)
