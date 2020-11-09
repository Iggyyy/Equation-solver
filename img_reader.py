import numpy as np

from matplotlib import image, pyplot as plt
from PIL import Image


img = image.imread('a.jpg')

print(img.dtype)
print(img.shape)

plt.imshow(img)
plt.show()

img = Image.open('a.jpg')

img = img.resize( (28,28), Image.ANTIALIAS)

plt.imshow(img)
plt.show()

