import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

# Define skin color range in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological transformations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # Calculate moments to find the centroid
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # x-coordinate of the centroid
            cy = int(M["m01"] / M["m00"])  # y-coordinate of the centroid
            print(f"Centroid Position: x={cx}, y={cy}")
        else:
            cx, cy = 0, 0

        # Draw the largest contour
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 3)

        # Calculate and display the bounding rectangle
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Print bounding box details
        print(f"Bounding Box: x={x}, y={y}, width={w}, height={h}")

    # Display the results
    cv2.imshow("Hand Detection", frame)
    cv2.imshow("Mask", mask)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()