import cv2
import numpy as np
from  matplotlib import pyplot as plt
from PIL import Image

#TODO upgrade check_internal to be more error-proof

def check_internal(x,y,w,h, it):
    """
    This function checks wheter passed recrangle is internal to any other 
    (except first which is the whole img)
    """
    for i in range(1, len(contours)):
        _cnt = contours[i]
        _x,_y,_w,_h = cv2.boundingRect(_cnt)
        if x >= _x and x <= _x + _w and y >= _y and y <= _y+_h and it != i:
            return True
    return False

def get_contours(img):
    """
    Processing pased image, tresholding, and finding contours.
    Returns contours and tresholded img.
    """
    print("Image shape and type: ", img.shape, img.dtype)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _ , thresh = cv2.threshold(imgray, 112,255,0)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(hierarchy)
    return contours, thresh

#Currently not using this one
def mask_image(img):
    """
    Returns masked image
    """
    b,g,r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_tresh = np.array([0,0,0])
    upper_tresh = np.array([180,255,30])

    mask = cv2.inRange(hsv, lower_tresh, upper_tresh)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

def extract_nums_from_img(contours, img, threshloded):
    """
    Extracts numbers from passed image.
    Returns list of cropped images with digits.
    """

    extracted_nums = []
    rec_params = []

    for i in range(1, len(contours)):
        """
        Finding contours and extracting single numbers
        """

        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)

        inc = 5 #var that increases rect boundaries

        if check_internal(x,y,w,h,i) == False:
            rec_params.append([x,y,w,h])

    rec_params = sorted(rec_params)    


    for x,y,w,h in rec_params:
        img = cv2.rectangle(img,(x-inc,y-inc),(x+w+inc,y+h+inc),(0,0,255),1)
        num = thresholded[y-inc:y+h+inc, x-inc:x+w+inc]
        extracted_nums.append(num)

    return extracted_nums



img = cv2.imread(r"Testing\nums.png")

contours, thresholded= get_contours(img)

extracted = extract_nums_from_img(contours, img, thresholded)

print("Number of digits found in image: ", len(extracted))

while(1):

    k = cv2.waitKey(5)
    if k == 13:
        break
    
    cv2.imshow('With boundaries: ', img)

cv2.destroyAllWindows()


#SAVING
for x in extracted:
    print(x.shape)


def reshape_to_square(image, desired_size):
    """
    Drawing additional cols and rows, so then reshaping will be less lossy.
    """
    while( image.shape[0] != image.shape[1]):
        height = image.shape[0]
        width = image.shape[1]

        if height > width:
            if width % 2 == 0:
                image = np.c_[image, np.full(height, 255)]
            else:
                image = np.c_[np.full(height, 255), image]

        else:
            if height % 2 == 0:
                image = np.r_[image, [np.full(width,255)]]
            else:
                image = np.r_[[np.full(width,255)], image]
    
    
    
    
    

    return image

for i in range(len(extracted)):
    extracted[i]  = reshape_to_square(extracted[i], 28)
    print(extracted[i][14])
    name = r"Testing\extr_" + str(i) + ".png"
    cv2.imwrite(name,extracted[i])

    im = cv2.imread(name)
    res = cv2.resize(im, dsize=(28,28), interpolation=cv2.INTER_CUBIC)

    res = (255-res)

    cv2.imwrite(name,res)

    
print("Succesfully saved images")

    