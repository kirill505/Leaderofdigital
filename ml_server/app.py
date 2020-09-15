import cv2
import time
import torch
import utils
import argparse
import numpy as np

from flask import Flask, render_template, Response
from PIL import Image
from models import gazenet
from mtcnn import FaceDetector

app = Flask(__name__,
            static_url_path='',
            static_folder='../frontend/static',
            template_folder='../frontend/templates')

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--weights', '-w', type=str, default='models/weights/gazenet.pth')
args = parser.parse_args()

print('Loading MobileFaceGaze model...')
device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu")
model = gazenet.GazeNet(device)

if not torch.cuda.is_available() and not args.cpu:
    print('Tried to load GPU but found none. Please check your environment')

state_dict = torch.load(args.weights, map_location=device)
model.load_state_dict(state_dict)
print('Model loaded using {} as device'.format(device))

model.eval()

# camera = cv2.VideoCapture('rtsp://192.168.1.64:8080/h264_pcm.sdp')
camera = cv2.VideoCapture('0')
# camera = cv2.VideoCapture('demo.mp4')

face_detector = FaceDetector(device=device)


# generate frame by frame from camera
def gen_frames():
    fps = 0
    frame_num = 0
    frame_samples = 6
    fps_timer = time.time()
    while True:
        # Capture frame-by-frame
        # success, frame = camera.read()  # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = frame.tobytes()

            frame = frame[:, :, ::-1]
            frame = cv2.flip(frame, 1)
            img_h, img_w, _ = np.shape(frame)
            frame_num += 1
            # Detect Faces
            display = frame.copy()
            faces, landmarks = face_detector.detect(Image.fromarray(frame))

            if len(faces) != 0:
                for f, lm in zip(faces, landmarks):
                    # Confidence check
                    if f[-1] > 0.98:
                        # Crop and normalize face Face
                        face, gaze_origin, m = utils.normalize_face(lm, frame)

                        # Predict gaze
                        with torch.no_grad():
                            gaze = model.get_gaze(face)
                            gaze = gaze[0].data.cpu()

                            # Draw results
                        display = cv2.circle(display, gaze_origin, 3, (0, 255, 0), -1)
                        display = utils.draw_gaze(display, gaze_origin, gaze, color=(255, 0, 0), thickness=2)

            # Calc FPS
            if frame_num == frame_samples:
                fps = time.time() - fps_timer
                fps = frame_samples / fps
                fps_timer = time.time()
                frame_num = 0
            display = cv2.putText(display, 'FPS: {:.2f}'.format(fps), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0),
                                  1, cv2.LINE_AA)
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            ret, res = cv2.imencode('.jpg', display)

            # concat frame one by one and show result
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + res.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
