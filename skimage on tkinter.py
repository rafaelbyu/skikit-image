from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.filters import threshold_sauvola, threshold_niblack, threshold_otsu


class VideoPlayer:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        self.root = Tk()
        self.root.title("Изображения")
        self.is_playing = False

        self.f_left = LabelFrame(self.root, text="Original and niblack")
        self.f_left.pack(side=LEFT, padx=15, pady=15)
        self.f_original_label = Label(self.f_left, text="изображение1")
        self.f_original_label.pack()
        self.f_niblack_label = Label(self.f_left, text="niblack")
        self.f_niblack_label.pack()

        self.f_right = LabelFrame(self.root, text="Sauvola and global")
        self.f_right.pack(side=RIGHT, padx=15, pady=15)
        self.f_sauvola_label = Label(self.f_right, text="изображение2")
        self.f_sauvola_label.pack()
        self.f_global_label = Label(self.f_right, text="global")
        self.f_global_label.pack()

        self.start_btn = Button(self.f_left, text="Start", command=self.start)
        self.start_btn.pack(padx=10, pady=10)

        self.stop_btn = Button(self.f_right, text="Stop", command=self.stop)
        self.stop_btn.pack(padx=10, pady=10)

    def play_video(self):
        while self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                break
            scale_percent = 25  # )percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1)

            frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            result_original = cv2.bitwise_and(frame_original, frame_original)
            image_original = Image.fromarray(result_original)
            photo_original = ImageTk.PhotoImage(image_original)
            self.f_original_label.config(image=photo_original)
            self.f_original_label.image = photo_original
            self.f_original_label.update()

            binary_sauvola = frame_gray > threshold_sauvola(frame_gray, window_size=25, k=0.9)
            binary_sauvola = (binary_sauvola * 255).astype(np.uint8)
            result_sauvola = cv2.bitwise_and(frame_gray, frame_gray, mask=binary_sauvola)
            image_sauvola = Image.fromarray(result_sauvola)
            photo_sauvola = ImageTk.PhotoImage(image_sauvola)
            self.f_sauvola_label.config(image=photo_sauvola)
            self.f_sauvola_label.image = photo_sauvola
            self.f_sauvola_label.update()

            binary_niblack = frame_gray > threshold_niblack(frame_gray, window_size=25, k=0.9)
            binary_niblack = (binary_niblack * 255).astype(np.uint8)
            result_niblack = cv2.bitwise_and(frame_gray, frame_gray, mask=binary_niblack)
            image_niblack = Image.fromarray(result_niblack)
            photo_niblack = ImageTk.PhotoImage(image_niblack)
            self.f_niblack_label.config(image=photo_niblack)
            self.f_niblack_label.image = photo_niblack
            self.f_niblack_label.update()

            binary_global = frame_gray > threshold_otsu(frame_gray)
            binary_global = (binary_global * 255).astype(np.uint8)
            result_global = cv2.bitwise_and(frame_gray, frame_gray, mask=binary_global)
            image_global = Image.fromarray(result_global)
            photo_global = ImageTk.PhotoImage(image_global)
            self.f_global_label.config(image=photo_global)
            self.f_global_label.image = photo_global
            self.f_global_label.update()

    def start(self):
        self.is_playing = True
        self.play_video()
        self.root.mainloop()

    def stop(self):
        self.is_playing = False


player = VideoPlayer("rtsp://visual:visualrav1@10.60.86.95:554/cam/realmonitor?channel=1&subtype=0")
player.start()
