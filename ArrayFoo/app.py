from flask import Flask, redirect, url_for, render_template, Response
from model import *


app = Flask(__name__)

# camera = cv.VideoCapture(
# r"C:\Users\Christian Paul\Downloads\WIN_20230503_14_14_13_Pro.mp4")
camera = cv.VideoCapture(0, cv.CAP_DSHOW)
camera.set(cv.CAP_PROP_FRAME_WIDTH, 4000)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 4000)


def gen_frames():
    #i = 0
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
                                     tf.cast(config['INPUT_shape']
                                             [0], tf.float32),
                                     pred[0], pred[1], pred[2])

            #cv.imwrite("boxed_frames/Frame" + str(i)+".jpg", frame.numpy())
            #i += 1
            #frame = tf.image.draw_bounding_boxes(frame[tf.newaxis, ...], XYXY_to_YXYX(pred[0][0][tf.newaxis, ...]), [[252.0, 3.0, 3.0], [18.0, 4.0, 217.0]])

            ret, buffer = cv.imencode('.jpg', frame.numpy())
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_boxframes():
    i = 0
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
                                     tf.cast(config['INPUT_shape']
                                             [0], tf.float32),
                                     pred[0], pred[1], pred[2])

            cv.imwrite("boxed_frames/FRAMES_2/Frame" +
                       str(i)+".jpg", frame.numpy())
            i += 1


def gen_vid():
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter('video1.mp4', fourcc, 30, (768, 768))

    for i in range(0, 2330):
        img = cv.imread("boxed_frames\FRAMES_2\Frame" + str(i) + ".jpg")
        print(f"Read frame {i}")
        video.write(img)
        print(f"Written frame {i}")

    video.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_frames')
def videos():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
