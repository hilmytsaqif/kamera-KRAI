import cv2
import numpy as np
import time

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Set FPS ke 30
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    # Hitung waktu awal frame
    start_time = time.time()
    
    # Baca frame dari kamera
    ret, frame = cap.read()
    
    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur gambar untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Deteksi lingkaran menggunakan HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=30, maxRadius=100)
    
    # Jika lingkaran terdeteksi
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Gambar lingkaran yang terdeteksi
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    # Tampilkan frame
    cv2.imshow("Circle Detection", frame)
    
    # Hitung FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    # Tampilkan FPS di console
    print(f"FPS: {fps:.2f}")
    
    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()