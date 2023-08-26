from __future__ import print_function

from PIL import Image, ImageFilter
import cv2
import cv2 as cv
import argparse

import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass


cv2.namedWindow("Config")
cv2.createTrackbar('history', "Config", 300, 1000, nothing)
cv2.createTrackbar('varThreshold', "Config", 5, 1000, nothing)
cv2.createTrackbar('setVarThresholdGen', "Config", 5, 255, nothing)
cv2.createTrackbar('setVarInit', "Config", 5, 100, nothing)
cv2.createTrackbar('setNMixtures', "Config", 100, 100, nothing)
cv2.createTrackbar('setVarMin', "Config", 1, 50, nothing)
cv2.createTrackbar('setVarMax', "Config", 10, 100, nothing)
# cv2.createTrackbar('setBackgroundRatio', "Config", 0.9, 1, nothing)

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
 OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='',
                    default='rtsp://visual:visualrav1@10.60.86.95:554/cam/realmonitor?channel=1&subtype=0')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(history=int(cv2.getTrackbarPos('history', 'Config')),
                                                varThreshold=int(cv2.getTrackbarPos('varThreshold', "Config")), detectShadows=False)
    backSub.setNMixtures(int(cv2.getTrackbarPos('setNMixtures', "Config")))
    backSub.setVarInit(int(cv2.getTrackbarPos('setVarInit', "Config")))
    backSub.setVarMin(int(cv2.getTrackbarPos('setVarMin', "Config")))
    backSub.setVarMax(int(cv2.getTrackbarPos('setVarMax', "Config")))
    backSub.setVarThresholdGen(int(cv2.getTrackbarPos('setVarThresholdGen', "Config")))
    backSub.setBackgroundRatio(0.9)

else:
    backSub = cv.createBackgroundSubtractorKNN(1000, 16, False)


capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)


while True:
    ret, frame = capture.read()
    if frame is None:
        break

    scale_percent = 90  # )percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    y = 310
    x = 500
    h = 340
    w = 490

    crop_frame = frame[y:y + h, x:x + w]
    # smol_image = crop_frame.crop((0, 0, 50, 50))
    # Grayscale
    def BGR2GRAY(img):
        # Grayscale
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    # Gabor Filter
    def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
        # get half size
        d = K_size // 2

        # prepare kernel
        gabor = np.zeros((K_size, K_size), dtype=np.float32)

        # each value
        for y in range(K_size):
            for x in range(K_size):
                # distance from center
                px = x - d
                py = y - d

                # degree -> radian
                theta = angle / 180. * np.pi

                # get kernel x
                _x = np.cos(theta) * px + np.sin(theta) * py

                # get kernel y
                _y = -np.sin(theta) * px + np.cos(theta) * py

                # fill kernel
                gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                    2 * np.pi * _x / Lambda + Psi)

        # kernel normalization
        gabor /= np.sum(np.abs(gabor))

        return gabor

    # Используйте фильтр Габора, чтобы воздействовать на изображение
    def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
        # get shape
        H, W = gray.shape

        # padding
        gray = np.pad(gray, (K_size // 2, K_size // 2), 'edge')

        # prepare out image
        out = np.zeros((H, W), dtype=np.float32)

        # get gabor filter
        gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)

        # filtering
        for y in range(H):
            for x in range(W):
                out[y, x] = np.sum(gray[y: y + K_size, x: x + K_size] * gabor)

        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)

        return out

    # Используйте 6 фильтров Габора с разными углами для извлечения деталей на изображении
    def Gabor_process(img):
        # get shape
        H, W, _ = img.shape

        # gray scale
        gray = BGR2GRAY(img).astype(np.float32)

        # define angle
        # As = [0, 45, 90, 135]
        As = [0, 30, 60, 90, 120, 150]

        # prepare pyplot
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

        out = np.zeros([H, W], dtype=np.float32)

        # each angle
        for i, A in enumerate(As):
            # gabor filtering
            _out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)

            # add gabor filtered image
            out += _out

        # scale normalization
        out = out / out.max() * 255
        out = out.astype(np.uint8)

        return out


    img = crop_frame.astype(np.float32)

    # gabor process
    out = Gabor_process(img)

    def gaussian_filter(img, K_size=3, sigma=1.3):

        if len(img.shape) == 3:
            H, W, C = img.shape

        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape

        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.cfloat)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(np.cfloat)

        K = np.zeros((K_size, K_size), dtype=np.cfloat)

        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

        K /= (2 * np.pi * sigma * sigma)
        K /= K.sum()
        tmp = out.copy()

        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

        out = np.clip(out, 0, 255)
        out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
        return out


    # out = gaussian_filter(crop_frame, K_size=3, sigma=1.3)

    crop_frame = cv2.GaussianBlur(crop_frame, (7, 7), 0)
    # crop_frame =Image.fromarray(crop_frame)

    fgMask = backSub.apply(crop_frame)

    ret, thresh_img = cv2.threshold(fgMask, 100, 255, cv2.THRESH_BINARY)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # fgMask = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    # mask = cv2.erode(fgMask, kernel, iterations=1)
    # mask = cv2.dilate(fgMask, kernel, iterations=1)

    contour, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = np.zeros(crop_frame.shape)

    cv2.drawContours(img_contours, contour, -1, (255, 255, 255), 2)

    cv.rectangle(crop_frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(crop_frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', crop_frame)
    cv.imshow('FG Mask', img_contours)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


cv2.waitKey(0)
cv2.destroyAllWindows()
