import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

matplotlib.rcParams['font.size'] = 20

cap = cv2.VideoCapture('GoogleChrom.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_sauvola.mp4', fourcc, 60.0, (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    binary_global = image > threshold_otsu(image)

    window_size = 25
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola

    binary_niblack = (binary_niblack * 255).astype(np.uint8)
    binary_sauvola = (binary_sauvola * 255).astype(np.uint8)
    binary_global = (binary_global * 255).astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    cv2.imshow('Original', cv2.bitwise_and(image, image))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Global Threshold')
    cv2.imshow('Global Threshold', cv2.bitwise_and(image, image, mask=binary_global))
    plt.axis('off')

    plt.subplot(2, 2, 3)
    cv2.imshow('Niblack Threshold', cv2.bitwise_and(image, image, mask=binary_niblack))
    plt.title('Niblack Threshold')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    cv2.imshow('Sauvola Threshold', cv2.bitwise_and(image, image, mask=binary_sauvola))
    plt.title('Sauvola Threshold')
    plt.axis('off')

    # plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
