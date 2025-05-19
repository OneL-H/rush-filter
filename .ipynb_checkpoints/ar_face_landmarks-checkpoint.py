import cv2
import dlib
import numpy 

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im)

    if len(rects) > 1:
        return None
    if len(rects) == 0:
        return None

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()

    if landmarks is None:
        return im

    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

cap = cv2.VideoCapture(0)

while True:   
    ret, frame = cap.read()   
    
    #Reduce image size by 75% to reduce processing time and improve framerates
    frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation = cv2.INTER_LINEAR)
    
    # flip image so that it's more mirror like
    frame = cv2.flip(frame, 1)

    image = frame
    landmarks = get_landmarks(image)
    image_with_landmarks = annotate_landmarks(image, landmarks)

    # 36 - 45 eyeglasses
    # 17 - 26 eyebrows
    # resize glasses - divide width between eyes 
    if landmarks is not None:  
        # get image
        glasses = cv2.imread('images/glasses.png', -1)

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
    
    # 17 - 26 eyebrows
    if landmarks is not None:  
        # get image
        fedora = cv2.imread('images/fedora.png', -1)

        # find "borders"
        fedora_width = landmarks[26][0, 0] - landmarks[17][0, 0]
        fedora_height = int(fedora.shape[1] * (fedora_width / fedora.shape[0]))

        # scale up
        new_wid = int(fedora_width * 1.2)
        new_hig = int(fedora_height * 0.4)
        fedora = cv2.resize(fedora, (new_wid, new_hig))  

        # find origin point (UL corner)
        x_origin = max(int(landmarks[17][0, 0]), 0)
        y_origin = max(int(landmarks[17][0, 1] - new_hig * 1.4), 0)
        # scuffed way to prevent errors from OOB

        # find end point (LR corner) by adding width and height
        x_end = x_origin + new_wid
        y_end = y_origin + new_hig
            
        # alpha channel shenanigans
        alpha_fedora = fedora[:,:,3] / 255.0
        alpha_large = 1.0 - alpha_fedora

        # add together
        for channel in range(0, 3):
            image_with_landmarks[y_origin:y_end, x_origin:x_end, channel] = (
                alpha_fedora * fedora[:, :, channel] + 
                alpha_large * image_with_landmarks[y_origin:y_end, x_origin:x_end, channel])


    cv2.imshow('Result', image_with_landmarks)
    cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows() 
