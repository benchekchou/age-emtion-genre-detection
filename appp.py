import streamlit as st
import cv2
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os
from pyngrok import ngrok
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
from PIL import Image, ImageColor
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Create necessary directories
os.makedirs(os.path.join('age', 'output'), exist_ok=True)
os.makedirs(os.path.join('emotion', 'output'), exist_ok=True)
os.makedirs(os.path.join('gendre', 'output'), exist_ok=True)


face_detector = MTCNN()
try:
    # Load models with better error handling
    emotion_path = os.path.join('emotion', 'output', 'emotion_model.keras')
    age_path = os.path.join('age', 'output', 'age_model_pretrained.h5')
    gender_path = os.path.join('gendre', 'output', 'gender_model.keras')
    
    if not os.path.exists(emotion_path):
        st.error(f"Emotion model not found at {emotion_path}")
        st.error("Please run train_emotion.py first")
        st.stop()
        
    if not os.path.exists(gender_path):
        st.error(f"Gender model not found at {gender_path}")
        st.error("Please run train_gender.py first")
        st.stop()

   
    emotion_model = load_model(emotion_path)
    age_model = load_model(age_path)
    gender_model = load_model(gender_path)
    
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.error("Please make sure you have trained all models first by running:")
    st.code("python age/train_age.py")
    st.code("python emotion/train_emotion.py")
    st.code("python gendre/train_gender.py")
    st.stop()

# Labels on Age, Gender and Emotion to be predicted
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges = ['positive', 'negative', 'neutral']

class_labels = emotion_ranges
gender_labels = gender_ranges
face_detector = MTCNN()


def predict_age_gender_emotion(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detect_faces(image)

    i = 0
    for face in faces:
        if len(face['box']) == 4:
            i = i + 1
            x, y, w, h = face['box']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Crop the face ROI from the grayscale image
            roi_gray = gray[y:y + h, x:x + w]

            # Resize the ROI to 48x48 pixels and apply histogram equalization
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray = cv2.equalizeHist(roi_gray)

            # Get the ROI ready for prediction by scaling it between 0 and 1
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Use the emotion model to predict the emotion label of the ROI
            output_emotion = class_labels[np.argmax(emotion_model.predict(roi))]

            # Use the gender model to predict the gender label of the ROI
            gender_img = cv2.resize(roi_gray, (100, 100), interpolation=cv2.INTER_AREA)
            gender_image_array = np.array(gender_img)
            gender_input = np.expand_dims(gender_image_array, axis=0)
            output_gender = gender_labels[np.argmax(gender_model.predict(gender_input))]

            # Use the age model to predict the age range of the ROI
            age_image = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_AREA)
            age_input = age_image.reshape(-1, 200, 200, 1)
            output_age = age_ranges[np.argmax(age_model.predict(age_input))]

            # Build the output string with the predicted age, gender, and emotion labels
            output_str = str(i) + ": " + output_gender + ', ' + output_age + ', ' + output_emotion

            # Draw a rectangle and the output string on the original image
            col = (0, 255, 0)
            cv2.putText(image, output_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), col, 2)
            print(output_str)

    # Return the annotated image with the predicted labels
    return image


def app():
    st.title("Age, Gender, and Emotion Recognition")

    # -------------Sidebar Section------------------------------------------------

    detection_mode = None

    with st.sidebar:

        title = '<p style="font-size: 25px;font-weight: 550;">Face Detection Settings</p>'
        st.markdown(title, unsafe_allow_html=True)

        # choose the mode for detection
        mode = st.radio("Choose Face Detection Mode", ('Image Upload',
                                                       'Webcam Image Capture',
                                                       'Webcam Realtime frame by frame'), index=0)
        if mode == 'Image Upload':
            detection_mode = mode
        elif mode == 'Video Upload':
            detection_mode = mode
        elif mode == "Webcam Image Capture":
            detection_mode = mode
        elif mode == 'Webcam Realtime frame by frame':
            detection_mode = mode
        elif mode == 'real time face detection':
            detection_mode = mode

    # -------------Image Upload Section-----------------------------------------------

    if detection_mode == "Image Upload":

        # Use the file_uploader function to capture an image from the user
        image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key=1)

        # If an image is uploaded, run the prediction model on the image
        if image_file is not None:
            # Convert the image to OpenCV format
            img = Image.open(image_file)
            img = np.array(img)

            # Run the prediction model on the image
            result_image = predict_age_gender_emotion(img)

            # Display the result image
            st.image(result_image, channels="BGR", use_column_width=True)

    # -------------Webcam Image Capture Section------------------------------------------------

    if detection_mode == "Webcam Image Capture":
        image = st.camera_input("Capture an Image from Webcam", disabled=False, key=1,
                                help="Make sure you have given webcam permission to the site")

        # If an image is captured, run the prediction model on the image
        if image is not None:
            img = Image.open(image)
            image = np.array(img) # the channel is RGB now you need to convert it from RGB to BGR

            # Call the predict_age_gender_emotion function to get the predicted labels
            predicted_image = predict_age_gender_emotion(image)

            # Display the predicted image with labels using the Streamlit image component
            st.image(predicted_image, channels="BGR")

    # -------------Webcam Realtime Section frame by frame------------------------------------------------

    if detection_mode == "Webcam Realtime frame by frame":
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Loop to capture frames and predict results in real-time
        while True:
            ret, frame = cap.read()
            labels = []
            frame = predict_age_gender_emotion(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display webcam video feed and predicted results
            st.image(frame, channels="BGR", use_column_width=True)

            # Press 'q' key to stop the webcam feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close all windows
        cap.release()
        cv2.destroyAllWindows()
# -------------Hide Streamlit Watermark-----------------------------------------------


if __name__ == '__main__':
    app()