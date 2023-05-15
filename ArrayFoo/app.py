from flask import Flask, redirect, url_for, render_template, Response, jsonify
from model import *
import time
import os
import requests  # for Telegram bot notifications
'''
import serial.tools.list_ports #for serial connection
'''
from datetime import datetime
import json
from flask import send_from_directory
import threading


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

@app.route('/data')
def get_data():

    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    json_data_file = str(dt_object.strftime("%Y-%m-%d"))

    with open('ArrayFoo\\static\\saved_json\\' + json_data_file + ".json", 'r') as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/images/<path:image_path>')
def get_image(image_path):
    return send_from_directory('static', image_path)


camera = cv.VideoCapture(0, cv.CAP_DSHOW)
camera.set(cv.CAP_PROP_FRAME_WIDTH, 4000)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 4000)

# Replace YOUR_BOT_TOKEN and CHAT_ID with your actual bot token and chat ID
bot_token = "6279869007:AAFisEDYs0KyOZblRHdl69JIpwHD75vFc4k"
chat_id = "-1001944030203"

def send_image_notif(image_path):
    # Send a photo
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {"photo": open(image_path, "rb")}
    data = {"chat_id": chat_id}
    response = requests.post(url, files=files, data=data)

    # Check the response
    if response.status_code == 200:
        print("Image sent successfully!")
    else:
        print(f"Failed to send image. Error code: {response.status_code}")


def gen_frames():
    while True:

        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv.flip(frame, 1)
            frame_resized = tf.image.resize_with_pad(
                frame, config['INPUT_shape'][0], config['INPUT_shape'][0])[tf.newaxis, ...]
            pred = predict(model, frame_resized)
            frame, predictions = draw_predictions(frame_resized, 
                                                  tf.cast(config['INPUT_shape'][0], tf.float32),
                                                  pred[0], pred[1], pred[2])

            if predictions != None:
                timestamp = time.time()
                dt_object = datetime.fromtimestamp(timestamp)
                folder_name = str(dt_object.strftime("%Y-%m-%d"))
                path_folder = "ArrayFoo\\static\\saved_frames\\" + folder_name
                strtime = str(dt_object.strftime("%Y-%m-%d_%H-%M-%S"))
                json_file = "ArrayFoo\\static\\saved_json\\" + folder_name + ".json"
                if not os.path.exists(path_folder):
                    os.makedirs(path_folder)

                file_name = path_folder + "\\Frame" + strtime + ".jpg"
                statistics = process_predictions(predictions, config['NUM_classes'])

                print("{}\tFound [{}] W  [{}] C".format(strtime, statistics[0], statistics[1]))

                data = {"Timestamp": strtime, "W": statistics[0], "C": statistics[1]}

                if (statistics[0] > statistics[1]):
                    command = "W"+'\r'
                elif (statistics[0] < statistics[1]):
                    command = "C"+'\r'

                if not os.path.exists(os.path.dirname(json_file)):
                    os.makedirs(json_file)

                if not os.path.exists(json_file):
                    with open(json_file, "w") as f:
                        json.dump([data], f)
                        print("JSON Data created")
                else:
                    with open(json_file, "r+") as f:
                        file_data = json.load(f)
                        file_data.append(data)
                        f.seek(0)
                        json.dump(file_data, f)
                        print("JSON Data Appended")

                '''
                cv.imwrite(file_name, frame.numpy())
                notification_thread = threading.Thread(target=send_image_notif, args=(file_name,))
                notification_thread.start()
                '''

            else:
                command = "NONE"+'\r'

            ret, buffer = cv.imencode('.jpg', frame.numpy())
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        '''serialInst.write(command.encode('utf-8'))'''
        # print(command)


def process_predictions(predictions, NUM_classes):
    statistics = [0 for _ in range(NUM_classes)]
    for pred in predictions:
        label = pred[0]
        statistics[label] += 1
    return statistics


@app.route('/')
def index():
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    json_data_file = str(dt_object.strftime("%Y-%m-%d"))
    json_file = 'ArrayFoo\\static\\saved_json\\' + json_data_file + ".json"
    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(json_file)

    return render_template('index.html')


@app.route('/video_frames')
def videos():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
