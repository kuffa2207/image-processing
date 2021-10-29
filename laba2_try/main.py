import sys
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from skimage import data
from skimage.morphology import watershed
from scipy import ndimage
from skimage.feature import peak_local_max

# открытие фото
photo = cv2.imread("8.jpg")
(b, g, r) = photo[0, 0]
print("Red: {}, Green: {}, Blue: {}" .format(r, g, b))
#Удаление шумов
b, g, r = cv2.split(photo)
rgb_photo = cv2.merge([r, g, b])
denoise = cv2.fastNlMeansDenoisingColored(photo, None, 9, 10, 8, 25)
b, g, r = cv2.split(denoise)
rgb_denoise = cv2.merge([r, g, b])





#Бинаризация
gray = cv2.cvtColor(rgb_denoise, cv2.COLOR_RGB2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# вычисление евклид расстояния до каждого пикселя
evk_dist = ndimage.distance_transform_edt(thresh)
#нахождение пик
local_max = peak_local_max(evk_dist, indices=False, min_distance=40, labels=thresh)
#маркеры
markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
#подлючение водораздела
labels = watershed(-evk_dist, markers, mask=thresh)

# перебор меток
total_area = 0
for label in np.unique(labels):
    if label == 0:
        continue

    # создание маски
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    # Нахождение контуров
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area += area
    cv2.drawContours(photo, [c], -1, (36,255,12), 4)

print(total_area)
cv2.imshow('Результат', photo)
cv2.waitKey()
