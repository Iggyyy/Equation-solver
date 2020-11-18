import numpy as np
import cv2
from  Testing import load_save_json
from matplotlib import pyplot as plt



def get_image_section(layer,row_from, row_to, col_from, col_to):
    section = layer[:,row_from:row_to,col_from:col_to]
    return section.reshape(-1,1,row_to-row_from, col_to-col_from)
def tanh(x):
    return np.tanh(x)

def pred(images):
    preds = []
    for i in range(len(images)):
        layer_0 = images[i:i+1]

        layer_0 = layer_0.reshape(layer_0.shape[0],28,28)
        layer_0.shape

        sects = list()
        for row_start in range(layer_0.shape[1]-kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0,
                                        row_start,
                                        row_start+kernel_rows,
                                        col_start,
                                        col_start+kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects,axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0]*es[1],-1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0],-1))
        layer_2 = np.dot(layer_1,weights_1_2)

        preds.append(layer_2)

 

    return np.array(preds)

          
images = []
imgs = []
for i in range(5):
    im = cv2.imread(r"Testing\extr_" + str(i) + ".png")
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgs.append(im)
    im = im.reshape(28*28) / 255
    images.append(im)

images = np.array(images)
kernels, weights_1_2, kernel_rows, kernel_cols = load_save_json.get_from_json(r"Models\model_2020-11-18.txt")


p = pred(images)
for x in p:
    print(int(np.argmax(x)))


"""
plt.imshow(imgs[0])
plt.show()

while(1):

    k = cv2.waitKey(5)
    if k == 13:
        break
    
    cv2.imshow("Predicted vs image: " + str(np.argmax(p[i])), cv2.resize(imgs[i], dsize=(184,184), interpolation=cv2.INTER_CUBIC))
"""

