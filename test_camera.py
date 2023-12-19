import cv2

# Try different indices if 2 does not work
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if ret:
    cv2.imshow("Test Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to capture frame")

cap.release()
