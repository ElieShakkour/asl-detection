import base64
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
predicted_character = ''  
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {1: 'A', 2: 'B',3:'C',4:'D',5:'',6:'E',7:'F',8:'G',9:'H'}

@app.route('/', methods=['GET'])
def get():
 return jsonify({'prediction': 1})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data')  # assuming data is passed as a JSON object
    # Decode the base64 string to obtain the image bytes
    image_bytes = base64.b64decode(data)
    prediction = make_prediction(image_bytes)
    return jsonify({'prediction': prediction or ''})  # Return an empty string if no gesture is detected


def make_prediction(data):
    data_aux = []
    x_ = []
    y_ = []
    global predicted_character
    # Assuming data contains the image in a suitable format
    frame = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_COLOR)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
  
        predicted_character = prediction[0]
        predicted_label = labels_dict[int(predicted_character)] 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    return predicted_label

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
