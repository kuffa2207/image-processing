import cv2 as cv
import numpy as np


#маска
def MakeMask(photo):
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    photo_er = cv.erode(photo, ker)
    photo_HSV = cv.cvtColor(photo_er, cv.COLOR_BGR2HSV)
    markers = np.zeros((photo.shape[0], photo.shape[1]), dtype="int32")
    markers[90:140, 90:140] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:255] = 1
    markers[236:255, 236:255] = 1
    leafsBGR = cv.watershed(photo_er, markers)
    healthy_part = cv.inRange(photo_HSV, (36, 25, 25), (86, 255, 255))
    ill_part = leafsBGR - healthy_part
    mask = np.zeros_like(photo, np.uint8)
    mask[leafsBGR > 1] = (255, 178, 255)
    mask[ill_part > 1] = (20, 100, 255)
    return mask
#удаление шума
def NonLocalMeans(photo_):
    b, g, r = cv.split(photo_)
    rgb_photo = cv.merge([r, g, b])
    denoise = cv.fastNlMeansDenoisingColored(photo_, None, 10, 10, 7, 21)
    b, g, r = cv.split(denoise)
    rgb_denoise = cv.merge([r, g, b])
    return rgb_denoise

def hystMedian(img): #Гистограмма выравнивания
    # split g,b,r
    g = img[:, :, 0]
    b = img[:, :, 1]
    r = img[:, :, 2]

    # calculate hist
    hist_r, bins_r = np.histogram(r, 256)
    hist_g, bins_g = np.histogram(g, 256)
    hist_b, bins_b = np.histogram(b, 256)

    # calculate cdf
    cdf_r = hist_r.cumsum()
    cdf_g = hist_g.cumsum()
    cdf_b = hist_b.cumsum()

    # remap cdf to [0,255]
    cdf_r = (cdf_r - cdf_r[0]) * 255 / (cdf_r[-1] - 1)
    cdf_r = cdf_r.astype(np.uint8)  # Transform from float64 back to unit8
    cdf_g = (cdf_g - cdf_g[0]) * 255 / (cdf_g[-1] - 1)
    cdf_g = cdf_g.astype(np.uint8)  # Transform from float64 back to unit8
    cdf_b = (cdf_b - cdf_b[0]) * 255 / (cdf_b[-1] - 1)
    cdf_b = cdf_b.astype(np.uint8)  # Transform from float64 back to unit8

    # get pixel by cdf table
    r2 = cdf_r[r]
    g2 = cdf_g[g]
    b2 = cdf_b[b]

    # merge g,b,r channel
    img2 = img.copy()
    img2[:, :, 0] = g2
    img2[:, :, 1] = b2
    img2[:, :, 2] = r2
    return img2

# вывод
photo = cv.imread("9.jpg")
hyst = hystMedian(photo)
denoise = NonLocalMeans(hyst)
mask = MakeMask(denoise)
b, g, r = cv.split(mask)
photo_ = cv.merge([r, g, b])
cv.imshow("before", photo)
cv.imshow("after", photo_)
k = cv.waitKey(0)
