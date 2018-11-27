import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

# define argparse to launch app from console
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file", nargs='?', const=1, default="test_zange.jpg")
args = vars(ap.parse_args())

# read image and copy
img = cv2.pyrDown(cv2.imread(args["image"], cv2.IMREAD_UNCHANGED))
h, w = img.shape[:2]
imgBasicCnt = img.copy()
imgAdaptCnt = img.copy()
unmodified = img.copy()
