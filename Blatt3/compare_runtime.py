import Blatt3
import cv2
from PIL import Image
import numpy as np
import timeit


image = np.array(Image.open('assets/image_biggest.jpg'))


elapsed_sobel_x = timeit.timeit(lambda: cv2.filter2D(image, -1, Blatt3.sobel_x), number=1)
elapsed_sobel_y = timeit.timeit(lambda: cv2.filter2D(image, -1, Blatt3.sobel_y), number=1)
elapsed_laplace = timeit.timeit(lambda: cv2.filter2D(image, -1, Blatt3.laplace_kernel), number=1)
elapsed_log = timeit.timeit(lambda: cv2.filter2D(image, -1, Blatt3.log_kernel), number=1)
elapsed_dog = timeit.timeit(lambda: cv2.filter2D(image, -1, Blatt3.dog_kernel), number=1)

print(f"Sobelx: {elapsed_sobel_x}")
print(f"Sobely: {elapsed_sobel_y}")
print(f"Laplace: {elapsed_laplace}")
print(f"Log: {elapsed_log}")
print(f"Dog: {elapsed_dog}")