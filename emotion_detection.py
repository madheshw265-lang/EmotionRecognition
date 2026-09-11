import cv2
import numpy as np
import tensorflow as tf

# Load the trained emotion recognition model
model = tf.keras.models.load_model(
    "models/emotion_model.h5",
    compile=False
)

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open the laptop's built-in camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started.")
print("Press Q to quit.")

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Process every detected face
    for (x, y, w, h) in faces:

        # Extract the face
        face = gray[y:y+h, x:x+w]

        # Resize to the model's required size
        face = cv2.resize(face, (48, 48))

        # Normalize pixel values
        face = face.astype("float32") / 255.0

        # Prepare image for the CNN
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Predict emotion
        prediction = model.predict(face, verbose=0)

        emotion_index = np.argmax(prediction[0])
        emotion = emotion_labels[emotion_index]
        confidence = float(prediction[0][emotion_index]) * 100

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Display emotion and confidence
        text = f"{emotion} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show the camera window
    cv2.imshow("Real-Time Facial Emotion Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
camera.release()
cv2.destroyAllWindows()