import cv2

# Open the laptop's built-in camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open the camera.")
    exit()

print("✅ Camera started!")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("❌ Could not read camera frame.")
        break

    cv2.imshow("Laptop Camera Test", frame)

    # Press Q to close
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()