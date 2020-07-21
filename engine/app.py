from flask import Flask, render_template, request, Response
import os
from deteccion import deteccion as dt
from imutils.video import VideoStream
import cv2
from tensorflow.keras.models import load_model
from camera_opencv import Camera
import numpy as np
app = Flask(__name__, template_folder=os.path.abspath("../gui/"))

prototxt = "modelo/deploy.prototxt"
weights =  "modelo/res10_300x300_ssd_iter_140000.caffemodel"
maskNetModel = "modelo/mask_detector.model"
faceNet = cv2.dnn.readNet(prototxt,weights)
maskNet = load_model(maskNetModel)

#camera = Camera()
#camera.run()


def genFaceOnly(camera):
    while True:
        frame = cv2.imdecode(np.frombuffer(camera.get_frame(), np.uint8), -1)

        locs = dt.detectFace(frame,faceNet)
        for box in locs:
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            color = (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color , 2)

        retval, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def gen(camera):
    while True:
        frame = cv2.imdecode(np.frombuffer(camera.get_frame(), np.uint8), -1)
        (locs, preds) = dt.detect_and_predict_mask(frame,faceNet,maskNet)
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mascarilla" if mask > withoutMask else "No Mascarilla"
            color = (0, 255, 0) if label == "Mascarilla" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        retval, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + buffer.tobytes() + b'\r\n')


    return r

@app.route("/video", methods=["GET"])
def video():

    return Response(genFaceOnly(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('dashboard.html')



app.run(port=4621,threaded=True)