import numpy as np

from matplotlib import image, pyplot as plt
from PIL import Image




img = Image.open('a.jpg')

img = img.resize( (28,28), Image.ANTIALIAS)

plt.imshow(img)
plt.show()

