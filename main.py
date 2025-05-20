from __future__ import annotations
import cv2
import numpy
import dlib
import mediapipe as mp
import numpy as np

import os
import sys
import time

import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

PREDICTOR_PATH = "assets\\shape_predictor_68_face_landmarks.dat"
fedora = cv2.imread('images/cowboy.png', -1)
round_glasses = cv2.imread('images/glasses.png', -1)
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return None
    if len(rects) == 0:
        return None

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def add_to_eyes(glasses, landmarks, image_with_landmarks):
    if landmarks is None:  
        return image_with_landmarks

    # find "borders"
    glasses_width = landmarks[45][0, 0] - landmarks[36][0, 0]
    glasses_height = landmarks[41][0, 1] - landmarks[37][0, 1]

    # scale up
    new_wid = int(glasses_width * 1.8)
    new_hig = int(glasses_height * 4)
    glasses = cv2.resize(glasses, (new_wid, new_hig))  

    # find origin point (UL corner)
    x_origin = int(landmarks[36][0, 0] - (new_wid - glasses_width) / 2)
    y_origin = int(landmarks[36][0, 1] - new_hig / 2)

    # find end point (LR corner) by adding width and height
    x_end = x_origin + new_wid
    y_end = y_origin + new_hig

    # alpha channel shenanigans
    alpha_glasses = glasses[:,:,3] / 255.0
    alpha_large = 1.0 - alpha_glasses

    # add together
    for channel in range(0, 3):
        image_with_landmarks[y_origin:y_end, x_origin:x_end, channel] = (
            alpha_glasses * glasses[:, :, channel] + 
            alpha_large * image_with_landmarks[y_origin:y_end, x_origin:x_end, channel])
    return image_with_landmarks
    
def add_hat(hat, landmarks, image_with_landmarks):
    if landmarks is None:  
        return image_with_landmarks

    # find "borders"
    hat_width = landmarks[26][0, 0] - landmarks[17][0, 0]
    hat_height = int(hat.shape[1] * (hat_width / hat.shape[0]))

    # scale up
    new_wid = int(hat_width * 1.2)
    new_hig = int(hat_height * 0.4)
    hat = cv2.resize(hat, (new_wid, new_hig))  

    # find "original" x & y origin
    x_origin = int(landmarks[17][0, 0])
    y_origin = int(landmarks[17][0, 1] - new_hig * 1.4)
    # scuffed way to prevent errors from OOB

    # left cut off
    if x_origin < 0:
        # cut off left to fit
        hat = hat[0:hat.shape[0], -x_origin:hat.shape[1]]
        # reduce width
        new_wid = hat.shape[1]
        # cap origin at 0
        x_origin = 0

    # right cut off
    if x_origin + new_wid > image_with_landmarks.shape[1]:
        # cut off right to fit
        cutoff = x_origin + new_wid - image_with_landmarks.shape[1]
        hat = hat[:, :hat.shape[1] - cutoff]
        # reduce width - reduce by excess: end of image - start of x + witdth
        new_wid = hat.shape[1]


    if y_origin < 0:
        # cut off top to fit
        hat = hat[-y_origin:hat.shape[0], 0:hat.shape[1]]
        # reduce hieght  
        new_hig = hat.shape[0]
        # cap origin at 0
        y_origin = 0

    # find end point (LR corner) by adding width and height
    x_end = x_origin + new_wid
    y_end = y_origin + new_hig

    hat = hat[0:(y_end - y_origin), 0:(x_end - x_origin)]
        
    # alpha channel shenanigans
    alpha_hat = hat[:,:,3] / 255.0
    alpha_large = 1.0 - alpha_hat

    # add together
    for channel in range(0, 3):
        image_with_landmarks[y_origin:y_end, x_origin:x_end, channel] = (
            alpha_hat * hat[:, :, channel] + 
            alpha_large * image_with_landmarks[y_origin:y_end, x_origin:x_end, channel])
        
    return image_with_landmarks

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Photo Booth")
        self.setGeometry(0, 0, 800, 500)

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit = QAction("Exit", self, triggered=qApp.quit)  # noqa: F821
        self.menu_file.addAction(exit)

        # Create a label for the display camera
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        # Thread in charge of updating the image
        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)

        # Model group
        self.group_model = QGroupBox("Apply Settings:")
        self.group_model.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        model_layout = QVBoxLayout()

        # background
        self.bg_dropdown = QComboBox()
        self.bg_dropdown.addItems(["None", "Space", "Desert"])
        model_layout.addWidget(QLabel("Background:"))
        model_layout.addWidget(self.bg_dropdown)
        self.bg_dropdown.currentTextChanged.connect(self.bg_changed)

        # hat
        self.hat_dropdown = QComboBox()
        self.hat_dropdown.addItems(["None", "Fedora", "Cowboy"])
        model_layout.addWidget(QLabel("Hat:"))
        model_layout.addWidget(self.hat_dropdown)
        self.hat_dropdown.currentTextChanged.connect(self.hat_changed)

        # glasses
        self.glasses_dropdown = QComboBox()
        self.glasses_dropdown.addItems(["None", "Square-ish", "Round"])
        model_layout.addWidget(QLabel("Glasses:"))
        model_layout.addWidget(self.glasses_dropdown)
        self.glasses_dropdown.currentTextChanged.connect(self.glasses_changed)
        
        self.group_model.setLayout(model_layout)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop/Close")
        self.button1.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.button2.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.group_model, 1)
        right_layout.addLayout(buttons_layout, 1)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(right_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.button1.clicked.connect(self.start)
        self.button2.clicked.connect(self.kill_thread)
        self.button2.setEnabled(False)

    @Slot()
    def hat_changed(self, new):
        self.th.set_hat(new)

    @Slot()
    def bg_changed(self, new):
        self.th.set_bg_image(new)

    @Slot()
    def glasses_changed(self, new):
        self.th.set_glasses(new)
    
    @Slot()
    def set_model(self, text):
        self.th.set_file(text)

    @Slot()
    def kill_thread(self):
        print("Finishing...")
        self.button2.setEnabled(False)
        self.button1.setEnabled(True)
        self.th.cap.release()
        cv2.destroyAllWindows()
        self.status = False
        self.th.terminate()
        # Give time for the thread to finish
        time.sleep(1)

    @Slot()
    def start(self):
        print("Starting...")
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        self.th.start()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        self.cap = True

        self.current_hat = None
        self.current_glasses = None
        self.bg_image = None

    def set_hat(self, new):
        print(new)
        if(new == "None"):
            self.current_hat = None
        elif(new == "Fedora"):
            self.current_hat = cv2.imread("images/fedora.png", cv2.IMREAD_UNCHANGED)
        elif(new == "Cowboy"):
            self.current_hat = cv2.imread("images/cowboy.png", cv2.IMREAD_UNCHANGED)

    def set_glasses(self, new):
        print(new)
        if(new == "None"):
            self.current_glasses = None
        elif(new == "Square-ish"):
            self.current_glasses = cv2.imread("images/glasses.png", cv2.IMREAD_UNCHANGED)
        elif(new == "Round"):
            self.current_glasses = cv2.imread("images/round_glasses.png", cv2.IMREAD_UNCHANGED)
    
    def set_bg_image(self, new):
        if(new == "None"):
           self.bg_image = None
        elif(new == "Space"):
            self.bg_image = cv2.imread("images/space.jpg", cv2.IMREAD_UNCHANGED)
        elif(new == "Desert"):
            self.bg_image = cv2.imread("images/desert.jpg", cv2.IMREAD_UNCHANGED)
        else:
            self.bg_image = None
        print(new)
    
    def run(self):
        self.cap = cv2.VideoCapture(0)
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
            
            while self.status:
                ret, frame = self.cap.read()
        
                if not ret:
                  #print("Ignoring empty camera frame.")
                  continue

                # flip image to act as "mirror"
                image = cv2.flip(frame, 1)

                if self.bg_image is not None:
                    bg_image = cv2.resize(self.bg_image, camera_size, interpolation = cv2.INTER_AREA)
                    # convert into rgb to make model work
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False

                    # process image with model (?)
                    results = selfie_segmentation.process(image)
                
                    image.flags.writeable = True

                    # convert back into the weirdo format
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                    # Draw selfie segmentation on the background image.
                    # To improve segmentation around boundaries, consider applying a joint
                    # bilateral filter to "results.segmentation_mask" with "image".
                    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    # The background can be customized.
                    #   a) Load an image (with the same width and height of the input image) to
                    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
                    #   b) Blur the input image by applying image filtering, e.g.,
                    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
                    mask = results.segmentation_mask

                    # blur the mask a bit to smooth out edges
                    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
                    alpha = np.expand_dims(blurred_mask, axis=-1)

                    # apply background using mask
                    image = (alpha * image + (1 - alpha) * bg_image).astype(np.uint8)
                
                landmarks = get_landmarks(image)
                #image = annotate_landmarks(image, landmarks)
                
                if landmarks is not None:
                    if self.current_hat is not None:
                        image = add_hat(self.current_hat, landmarks, image)
                    if self.current_glasses is not None:
                        image = add_to_eyes(self.current_glasses, landmarks, image)

                # reprocess into rgb because cv2 is weird like that
                output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = output_image.shape
                img = QImage(output_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
                scaled_img = img.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
    
                # Emit signal
                self.updateFrame.emit(scaled_img)
        sys.exit(-1)

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())