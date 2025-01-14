import cv2
import numpy as np

color_ranges = {
    "red": ([0, 120, 70], [10, 255, 255], (0, 0, 255)),
    "green": ([36, 100, 100], [86, 255, 255], (0, 255, 0)),
    "blue": ([94, 80, 2], [126, 255, 255], (255, 0, 0)),
    "yellow": ([20, 100, 100], [40, 255, 255], (0, 255, 255)),
    "orange": ([5, 150, 150], [15, 255, 255], (0, 165, 255)),
    "purple": ([125, 50, 50], [175, 255, 255], (255, 0, 255)),
    "pink": ([140, 50, 50], [170, 255, 255], (203, 192, 255)),
    "cyan": ([85, 100, 100], [95, 255, 255], (255, 255, 0)),
    "black": ([0, 0, 0], [180, 255, 50], (0, 0, 0)),
    "white": ([0, 0, 200], [180, 30, 255], (255, 255, 255)),
}

def white_balance(img):
    img = np.float32(img)
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg = (avg_b + avg_g + avg_r) / 3
    img[:, :, 0] = img[:, :, 0] * (avg / avg_b)
    img[:, :, 1] = img[:, :, 1] * (avg / avg_g)
    img[:, :, 2] = img[:, :, 2] * (avg / avg_r)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    frame = white_balance(frame)
    cv2.imshow("White Balanced", frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color, (lower, upper, highlight_color) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # This line breaks my computer but it looks cool so just know you have to control-c to exit lol
        # cv2.imshow(f"{color.capitalize()} Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Reduce noise
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(frame, [contour], -1, highlight_color, thickness=3)

    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
