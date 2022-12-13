from threading import Thread
import cv2
from PIL import Image, ImageTk
import tkinter as tk


class Camera_stream:

    def __init__(self, src=0, panel=tk.Label):
        self.camera = cv2.VideoCapture(src)
        codec       = 1196444237.0 # MJPEG
        self.camera.set(cv2.CAP_PROP_FOURCC, codec)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.width  = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.crop_size  = 512
        self.start_x    = (self.width - self.crop_size) / 2
        self.end_x      = self.width - ((self.width - self.crop_size) / 2)
        self.start_y    = (self.height - self.crop_size) / 2
        self.end_y      = self.height - ((self.height - self.crop_size) / 2)

        self.stopped    = True
        self.grabbed    = False
        self.frame      = None
        self.tk_panel   = panel

        # Automaticly starts camera
        self.start()

    def capture(self):
        cv2.imwrite("./input.jpg", self.frame)

    def start(self):
        print("Camera width: ", self.width)
        print("Camera height: ", self.height)
        self.stopped = False
        (self.grabbed, self.frame) = self.camera.read()
        Thread(target=self.__show, args=()).start()
        return self

    def __show(self):
        if not self.grabbed:
            self.stop()
        else:
            while not self.stopped:
                (self.grabbed, self.frame) = self.camera.read()
                cv2.waitKey(1)
        print("camera stopped")

        # if not self.grabbed:
        #     self.stop()
        # else:
        #     while not self.stopped:
        #             (self.grabbed, self.frame) = self.camera.read()
        #             img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        #             img_pil = Image.fromarray(img)
        #             img_resized = img_pil.resize(
        #                                 (self.crop_size, self.crop_size), 
        #                                 box=(self.start_x, self.start_y, self.end_x, self.end_y))
        #             output_image = ImageTk.PhotoImage(img_resized)
        #             self.tk_panel.configure(image=output_image)
        #             self.tk_panel.img = output_image
        #             cv2.waitKey(1)
        # self.tk_panel.configure(image='')
        # print("camera stopped")
        
    def stop(self):
        self.stopped = True

    def release(self):
        self.stop()
        cv2.destroyAllWindows()
        self.camera.release()

    def set_tk_panel(self, panel=tk.Label):
        self.tk_panel = panel