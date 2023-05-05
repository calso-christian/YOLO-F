from flask import Flask, redirect, url_for, render_template, Response
from model import *
import time
import requests #for Telegram bot notifications
#import serial.tools.list_ports #for serial connection
from datetime import datetime


app = Flask(__name__)

'''
# PySerial communication with Arduino
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
portsList = []
for onePort in ports:
    portsList.append(str(onePort))
    print(str(onePort))
# input COM port number
val = input("Select Port: COM")
for x in range(0,len(portsList)):
    if portsList[x].startswith("COM"+str(val)):
        portVar = "COM" + str(val)
        print(portVar)
serialInst.baudrate = 115200
serialInst.port = portVar
serialInst.open()
'''
# camera = cv.VideoCapture(
# r"C:\Users\Christian Paul\Downloads\WIN_20230503_14_14_13_Pro.mp4")
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
            pred = model.predict(frame_resized, verbose=0)

            frame, predictions = draw_predictions(frame_resized,
                                     tf.cast(config['INPUT_shape'][0], tf.float32),
                                     pred[0], pred[1], pred[2])
            #command = "NONE"+'\r'

            if tf.reduce_any(pred[0][..., 0] > 0.0):
                timestamp = time.time()
                dt_object = datetime.fromtimestamp(timestamp)
                strtime = str(dt_object.strftime("%Y-%m-%d_%H:%M:%S"))
                statistics = process_predictions(predictions, config['NUM_classes'])

                print("{}\tFound [{}] W  [{}] C".format(strtime, statistics[0], statistics[1]))

                cv.imwrite("ArrayFoo\testfolder\Frame" + ".jpg", frame.numpy())
                command = "GESTURES"+'\r'
            
            else:
                command = None
            
            ret, buffer = cv.imencode('.jpg', frame.numpy())
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        #serialInst.write(command.encode('utf-8'))
          

def process_predictions(predictions, NUM_classes):
    statistics = [0 for _ in range(NUM_classes)]
    for pred in predictions:
        label = pred[0]
        statistics[label] += 1
        ## PERFORM ARDUINO PROCESSING PER LABEL HERE
    return statistics

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
