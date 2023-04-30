from flask import Flask, redirect, url_for, render_template, Response
import cv2
from model import *


app = Flask(__name__)

camera = cv2.VideoCapture(0)


def gen_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break
        else:

            frame = tf.image.resize_with_pad(frame, 768, 768)
            pred = model.predict(frame[tf.newaxis, ...])
            #frame = draw_predictions(frame, pred[0], pred[1], pred[2])
            frame = tf.image.draw_bounding_boxes(
                frame[tf.newaxis, ...], XYXY_to_YXYX(pred[0][0][tf.newaxis, ...]), [[252.0, 3.0, 3.0], [18.0, 4.0, 217.0]])
           
            ret, buffer = cv2.imencode('.jpg', frame.numpy()[0])

            #ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_frames')
def videos():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
