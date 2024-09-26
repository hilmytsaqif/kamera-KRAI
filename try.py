import cv2
import numpy as np
import time


cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FPS, 30)

while True:

    start_time = time.time()
    

    ret, frame = cap.read()
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=30, maxRadius=100)
    
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
       
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    
    cv2.imshow("Circle Detection", frame)
    
   
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
   
    print(f"FPS: {fps:.2f}")
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()