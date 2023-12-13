import numpy as np
import cv2

''' 
This file contains pre-processing function.
'''

minValue = 70


def func(path):
    frame = cv2.imread(path)

    hand_region = extract_hand_region(frame)

    return hand_region


def extract_hand_region(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    # Apply adaptive thresholding
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply Otsu's thresholding
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return res
