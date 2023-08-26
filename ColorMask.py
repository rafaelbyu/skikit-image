from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
 OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='', default='GoogleChrom.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
elif args.algo == 'KNN':
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)


def segment_fish(image):

    # Конвертация изображения в HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Установка оранжевого диапазона
    light_orange = (1, 190, 200)
    dark_orange = (18, 255, 255)

    # Применение оранжевой маски
    mask = cv.inRange(hsv_image, light_orange, dark_orange)

    # Установка белого диапазона
    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)

    # Применение белой маски
    mask_white = cv.inRange(hsv_image, light_white, dark_white)

    # Объединение двух масок
    final_mask = mask + mask_white
    result = cv.bitwise_and(image, image, mask=final_mask)

    # Сглаживание сегментации с помощью размытия
    # blur = cv.GaussianBlur(result, (7, 7), 0)
    return result


while True:
    ret, frame = capture.read()
    if frame is None:
        break

    scale_percent = 30  # )percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

    fgMask = segment_fish(frame) #backSub.apply()

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
