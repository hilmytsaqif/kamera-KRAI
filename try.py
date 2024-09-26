from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

source_window = 'Image'
maxTrackbar = 25
rng.seed(12345)

# Jadikan src dan src_gray sebagai global
global src, src_gray

def goodFeaturesToTrack_Demo(val):
    global src, src_gray  # Akses variabel global

    maxCorners = max(val, 1)  # Pastikan minimal 1 corner terdeteksi

    # Parameter untuk algoritma Shi-Tomasi
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04

    # Salin gambar sumber
    copy = np.copy(src)

    # Terapkan deteksi sudut
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)

    if corners is not None:
        # Gambar sudut yang terdeteksi
        print('** Jumlah sudut yang terdeteksi:', corners.shape[0])
        radius = 4
        for i in range(corners.shape[0]):
            cv.circle(copy, (int(corners[i,0,0]), int(corners[i,0,1])), radius, 
                       (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)

        # Tampilkan hasil
        cv.imshow(source_window, copy)

        # Set parameter untuk menemukan sudut yang diperbaiki
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)

        # Hitung lokasi sudut yang telah diperbaiki
        corners = cv.cornerSubPix(src_gray, corners, winSize, zeroZone, criteria)

        # Tampilkan hasilnya
        for i in range(corners.shape[0]):
            print(" -- Sudut yang diperbaiki [", i, "]  (", corners[i,0,0], ",", corners[i,0,1], ")")
    else:
        print("Tidak ada sudut yang ditemukan.")

# Load gambar sumber dan konversi menjadi grayscale
parser = argparse.ArgumentParser(description='Kode untuk tutorial detektor sudut Shi-Tomasi.')
parser.add_argument('--input', help='Path ke gambar input.', default='pic3.png')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Tidak dapat membuka atau menemukan gambar:', args.input)
    exit(0)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Buat jendela dan trackbar
cv.namedWindow(source_window)
maxCorners = 10  # threshold awal
cv.createTrackbar('Threshold: ', source_window, maxCorners, maxTrackbar, goodFeaturesToTrack_Demo)
cv.imshow(source_window, src)
goodFeaturesToTrack_Demo(maxCorners)

cv.waitKey()
cv.destroyAllWindows()  # Tutup semua jendela OpenCV
