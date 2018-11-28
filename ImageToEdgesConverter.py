import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

# define argparse to launch app from console
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file", nargs='?', const=1, default="emoji.png")
args = vars(ap.parse_args())

# read image and copy
img = cv2.pyrDown(cv2.imread(args["image"], cv2.IMREAD_UNCHANGED))
h, w = img.shape[:2]
imgBasicCnt = img.copy()
imgAdaptCnt = img.copy()
unmodified = img.copy()

''' currently not in use
# downsampling via gaußian pyramid
rows, cols, _channels = map(int, unmodified.shape)
pyredImage = cv2.pyrDown(unmodified.copy(), (cols, rows))
pyredUp = cv2.pyrUp(pyredImage.copy(), (cols, rows))

# downsampling via interpolation
resized = cv2.resize(unmodified.copy(), (cols // 2, rows // 2), 0, 0, interpolation=cv2.INTER_AREA)
upsized = cv2.resize(resized.copy(), (cols, rows), 0,0, interpolation=cv2.INTER_AREA)
'''

# grayscale & blurr image
gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
blurredImg = cv2.GaussianBlur(gray.copy(), (7, 7), 0)

# basic threshhold
ret, threshBasic = cv2.threshold(blurredImg, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# adaptive threshhold
threshAdapt = cv2.adaptiveThreshold(blurredImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

# kernel and number of iterations
kernel = np.ones((9, 9), np.uint8)
num_iters = 3

# erosion and closing on thresh images
threshBasic = cv2.morphologyEx(threshBasic, cv2.MORPH_CLOSE, kernel, num_iters)
#threshBasic = cv2.morphologyEx(threshBasic, cv2.MORPH_DILATE, kernel, num_iters)
#threshBasic = cv2.morphologyEx(threshBasic, cv2.MORPH_GRADIENT, kernel, num_iters)

threshAdapt = cv2.morphologyEx(threshAdapt, cv2.MORPH_CLOSE, kernel, num_iters)
#threshAdapt = cv2.morphologyEx(threshAdapt, cv2.MORPH_ERODE, kernel, num_iters)
#threshAdapt = cv2.morphologyEx(threshAdapt, cv2.MORPH_GRADIENT, kernel, num_iters)

image, contours, hier = cv2.findContours(
    threshBasic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imageAdapt, contoursAdapt, hier = cv2.findContours(
    threshAdapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ShapeMatch both contours to indicate similarity
ret = cv2.matchShapes(contours[0],contoursAdapt[0],1,0.0)
print(ret)

for c in contours:
    # find minimum area and calculate coordinates of the minimum area rectangle
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

for cont in contoursAdapt:
    epsilon = 0.001 * cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, epsilon, True)

# draw basic contours
contImgBasic = np.zeros((h, w, 3), np.uint8)
contImgBasic.fill(255)
cv2.drawContours(contImgBasic, contours, -1, (255, 89, 0), 2)

# draw adaptive contours
contImgAdapt = np.zeros((h, w, 3), np.uint8)
contImgAdapt.fill(255)
cv2.drawContours(contImgAdapt, [approx], -1, (255, 89, 0), 2)

# draw contours on top of input image
cv2.drawContours(imgBasicCnt, contours, -1, (255, 89, 0), 2)
cv2.drawContours(imgAdaptCnt, [approx], -1, (255, 89, 0), 2)

# concatenate baseImage and both contourLines into 1 image
numpy_vertical_concat01 = np.concatenate((unmodified, contImgBasic), axis=1)
numpy_vertical_concat01 = np.concatenate(
    (numpy_vertical_concat01, contImgAdapt), axis=1)
numpy_vertical_concat02 = np.concatenate((img, imgBasicCnt), axis=1)
numpy_vertical_concat02 = np.concatenate(
    (numpy_vertical_concat02, imgAdaptCnt), axis=1)
numpy_vertical_concat03 = np.concatenate((threshBasic, threshAdapt), axis=1)
entireImg = np.concatenate(
    (numpy_vertical_concat01, numpy_vertical_concat02), axis=0)

# display the images
cv2.namedWindow("Contour Comparison", cv2.WINDOW_NORMAL)
cv2.imshow("Contour Comparison", entireImg)
cv2.imshow("thresh Comparison", numpy_vertical_concat03)
cv2.waitKey()
cv2.destroyAllWindows()
