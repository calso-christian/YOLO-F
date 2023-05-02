from flask import Flask, redirect, url_for, render_template, Response
from model import *

app = Flask(__name__)

camera = cv.VideoCapture(0, cv.CAP_DSHOW)
camera.set(cv.CAP_PROP_FRAME_WIDTH, 4000)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 4000)

def gen_frames():

    while True:
        
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv.flip(frame, 1)
            frame_resized = tf.image.resize_with_pad(
                frame, config['INPUT_shape'][0], config['INPUT_shape'][0])[tf.newaxis, ...]
            pred = model.predict(frame_resized)

            frame = draw_predictions(frame_resized,
                                     tf.cast(config['INPUT_shape'][0], tf.float32),
                                     pred[0], pred[1], pred[2])

            #frame = tf.image.draw_bounding_boxes(frame[tf.newaxis, ...], XYXY_to_YXYX(pred[0][0][tf.newaxis, ...]), [[252.0, 3.0, 3.0], [18.0, 4.0, 217.0]])
           
            ret, buffer = cv.imencode('.jpg', frame.numpy())
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
