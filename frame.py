#
import cv2

# Open the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera")
    exit()

while True:
    for i in range(100): 
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            print("Failed to read the frame")
            break

        # Display the frame
        cv2.imshow("Camera", frame)

    # Wait for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows 
cap.release()
cv2.destroyAllWindows()
