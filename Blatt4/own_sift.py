import cv2 as cv
import numpy as np
import SIFT_KeyPoint as SK

def create_scale_space(image, sp):

    deltas = [sp.delta_min * 2**i for i in range(sp.n_oct)]
    # delta_oct / delta_min * sigma_mind * 2**scale/n_spo
    #k = 2**1/n_spo, sigma = 2 s_min * k**scale
    sigmas = [[sp.sigma_min * pow(2.0, scale / sp.n_spo + oct) for scale in range(sp.n_spo + 3)] for oct in range(sp.n_oct)]
    scale_space = []
    
    for octave in range(sp.n_oct):
        octave_images = []
        resized_image = cv.resize(image, (0, 0), fx=1.0 / deltas[octave], fy=1.0 / deltas[octave], interpolation=1)
        for scale in range(sp.n_spo + 3): 
            sigma_scale = sigmas[octave][scale] / deltas[octave]
            
            blurred_image = cv.GaussianBlur(resized_image, (0, 0), sigma_scale)
            octave_images.append(blurred_image)
        scale_space.append(octave_images)
        
    return scale_space, deltas, sigmas

def create_dogs(scale_space, sp):
    all_dogs = []
    for octave_images in scale_space:
        octave_dogs = []
        for i in range(1, len(octave_images)):
            dog = cv.subtract(octave_images[i], octave_images[i - 1])
            octave_dogs.append(dog)
        all_dogs.append(octave_dogs)
    return all_dogs

def find_discrete_extremas(all_dogs, sp, sigmas, deltas):
    keypoints = []
    for oct, octave_dogs in enumerate(all_dogs):
        for sc in range(1, len(octave_dogs) - 1):
            prev = octave_dogs[sc - 1]
            curr = octave_dogs[sc]
            next = octave_dogs[sc + 1]
            for i in range(1, curr.shape[0] - 1):
                for j in range(1, curr.shape[1] - 1):
                    value = curr[i, j]
                    if np.abs(value) > sp.C_DoG:  # Contrast Dog
                        neighborhood = np.concatenate((
                            prev[i-1:i+2, j-1:j+2].flatten(),
                            np.delete(curr[i-1:i+2, j-1:j+2].flatten(), 4),
                            next[i-1:i+2, j-1:j+2].flatten()
                        ))
                        
                        if np.max(neighborhood) < value or np.min(neighborhood) > value:
                            kp = SK.SIFT_KeyPoint(oct, sc, i, j, sigmas[oct][sc])
                            keypoints.append(kp)
    return keypoints