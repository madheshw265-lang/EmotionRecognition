import cv2

# Load OpenCV's face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

print("Camera started!")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    # Convert camera image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Draw rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()