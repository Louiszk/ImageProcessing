import matplotlib.pyplot as plt
import numpy as np

an_image = plt.imread('test.png')
# plt.imshow(an_image)

weights = [0.2989, 0.5870, 0.1140]
weights2 = [1.0, 0.5870, 0.1140]
weights3 = [-1.0 , 1.0, 1.0]

grayscale_image = np.dot(an_image[..., :3], weights3)
plt.imshow(grayscale_image, cmap=plt.get_cmap("gray"))
plt.show()