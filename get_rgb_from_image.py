import numpy as np
import cv2 as cv
import math
import time
from sklearn.cluster import KMeans
from collections import Counter

# Absolute Path
img = ''

# container pixel
widthArr = []
heightArr = []

# container coordinate pixel
coordinateMain = np.zeros((5, 2))

# container GBR pixel
GBRPixel = np.zeros((5, 3))
ordered_color = np.zeros((5, 3))

# container gradient

arr1_GBRPixel0 = []
arr1_GBRPixel1 = []
arr1_GBRPixel2 = []
arr1_GBRPixel0new = []
arr1_GBRPixel1new = []
arr1_GBRPixel2new = []

arr2_GBRPixel0 = []
arr2_GBRPixel1 = []
arr2_GBRPixel2 = []
arr2_GBRPixel0new = []
arr2_GBRPixel1new = []
arr2_GBRPixel2new = []

arr3_GBRPixel0 = []
arr3_GBRPixel1 = []
arr3_GBRPixel2 = []
arr3_GBRPixel0new = []
arr3_GBRPixel1new = []
arr3_GBRPixel2new = []

arr4_GBRPixel0 = []
arr4_GBRPixel1 = []
arr4_GBRPixel2 = []

arr5_GBRPixel0 = []
arr5_GBRPixel1 = []
arr5_GBRPixel2 = []


def main(img):
    thisImg = cv.imread(img)
    print(thisImg)
    if thisImg.shape[0] >= 1080 and thisImg.shape[1] >= 1080:
        thisImgs = cv.resize(thisImg, (1080, 1080))
    elif thisImg.shape[0] >= 512 or thisImg.shape[1] >= 512 and thisImg.shape[0] <= 1080 and thisImg.shape[1] <= 1080:
        thisImgs = cv.resize(thisImg, (512, 512))
    elif thisImg.shape[0] >= 216 or thisImg.shape[1] >= 512:
        thisImgs = cv.resize(thisImg, (216, 216))
    print(thisImgs.shape)
    getDimension(thisImgs)


def getDimension(img):
    print("[+] Get Dimension from Image")
    height = img.shape[0]  # height scale
    width = img.shape[1]  # width scale
    getPixel(height, width)


def getPixel(height, width):
    print("[+] Get Pixel from Image")
    for x in range(1, 6):
        # input height scale / x (such as 1080/5, 1080/4, etc)
        widthArr.append(math.floor(height/x))

    for y in range(1, 6):
        # input width scale / x (such as 1080/5, 1080/4, etc)
        heightArr.append(math.floor(width/y))

    appendCoordinate()


def appendCoordinate():
    print("[+] Insert Coordinate Pixel to Array")
    # pixel is (x,y)
    for x in range(5):
        for y in range(1):
            # coordinate[x][0] (x axis)
            coordinateMain[x][y] = widthArr[x]

    for x in range(5):
        for y in range(1, 2):
            # coordinate[x][1] (y axis)
            coordinateMain[x][y] = heightArr[x]

    getGBRFromPixel()


def getGBRFromPixel():
    print("[+] Get GRB Value from Pixel Image")
    main_image = cv.imread(img)
    for x in range(5):  # dimensi coordinat
        for y in range(2):  # dimensi coordinat
            # main_image[1080,1080], etc
            color = main_image[int(coordinateMain[x][y])-1,
                               int(coordinateMain[x][y])-1]
            GBRPixel[x][y] = color[y]  # insert Green, Blue color to array
        for z in range(2, 3):
            color = main_image[int(coordinateMain[x][y])-1,
                               int(coordinateMain[x][y])-1]
            GBRPixel[x][z] = color[z]  # insert Red color to array

    makeCanvas()


def rgbToHex(r, g, b):
    return '#%02x%02x%02x' % (r, g, b)


def getGradientPoint():
    # GRADINET 1
    for x in range(255):
        if (255 in arr1_GBRPixel0):
            arr1_GBRPixel0.append(255)
        else:
            arr1_GBRPixel0.append(int(GBRPixel[0][0])+x)

    for x in range(255):
        if (255 in arr1_GBRPixel1):
            arr1_GBRPixel1.append(255)
        else:
            arr1_GBRPixel1.append(int(GBRPixel[0][1])+x)

    for x in range(255):
        if (255 in arr1_GBRPixel2):
            arr1_GBRPixel2.append(255)
        else:
            arr1_GBRPixel2.append(int(GBRPixel[0][2])+x)

    # GRADIENT 2
    for x in range(255):
        if (255 in arr2_GBRPixel0):
            arr2_GBRPixel0.append(255)
        else:
            arr2_GBRPixel0.append(int(GBRPixel[1][0])+x)

    for x in range(255):
        if (255 in arr2_GBRPixel1):
            arr2_GBRPixel1.append(255)
        else:
            arr2_GBRPixel1.append(int(GBRPixel[1][1])+x)

    for x in range(255):
        if (255 in arr2_GBRPixel2):
            arr2_GBRPixel2.append(255)
        else:
            arr2_GBRPixel2.append(int(GBRPixel[1][2])+x)

    # GRADIENT 3
    for x in range(255):
        if (255 in arr3_GBRPixel0):
            arr3_GBRPixel0.append(255)
        else:
            arr3_GBRPixel0.append(int(GBRPixel[2][0])+x)

    for x in range(255):
        if (255 in arr3_GBRPixel1):
            arr3_GBRPixel1.append(255)
        else:
            arr3_GBRPixel1.append(int(GBRPixel[2][1])+x)

    for x in range(255):
        if (255 in arr3_GBRPixel2):
            arr3_GBRPixel2.append(255)
        else:
            arr3_GBRPixel2.append(int(GBRPixel[2][2])+x)

    # GRADIENT 4
    for x in range(255):
        if (255 in arr4_GBRPixel0):
            arr4_GBRPixel0.append(255)
        else:
            arr4_GBRPixel0.append(int(GBRPixel[3][0])+x)

    for x in range(255):
        if (255 in arr4_GBRPixel1):
            arr4_GBRPixel1.append(255)
        else:
            arr4_GBRPixel1.append(int(GBRPixel[3][1])+x)

    for x in range(255):
        if (255 in arr4_GBRPixel2):
            arr4_GBRPixel2.append(255)
        else:
            arr4_GBRPixel2.append(int(GBRPixel[3][2])+x)

    # GRADIENT 5
    for x in range(255):
        if (255 in arr5_GBRPixel0):
            arr5_GBRPixel0.append(255)
        else:
            arr5_GBRPixel0.append(int(GBRPixel[4][0])+x)

    for x in range(255):
        if (255 in arr5_GBRPixel1):
            arr5_GBRPixel1.append(255)
        else:
            arr5_GBRPixel1.append(int(GBRPixel[4][1])+x)

    for x in range(255):
        if (255 in arr5_GBRPixel2):
            arr5_GBRPixel2.append(255)
        else:
            arr5_GBRPixel2.append(int(GBRPixel[4][2])+x)


def clearGradient():
    arr1_GBRPixel0.clear()
    arr1_GBRPixel1.clear()
    arr1_GBRPixel2.clear()

    arr2_GBRPixel0.clear()
    arr2_GBRPixel1.clear()
    arr2_GBRPixel2.clear()

    arr3_GBRPixel0.clear()
    arr3_GBRPixel1.clear()
    arr3_GBRPixel2.clear()

    arr4_GBRPixel0.clear()
    arr4_GBRPixel1.clear()
    arr4_GBRPixel2.clear()

    arr5_GBRPixel0.clear()
    arr5_GBRPixel1.clear()
    arr5_GBRPixel2.clear()


def makeCanvas():
    print("[+] Make Canvas and Show")
    # adittional canvas and text
    w = 765
    h = 500
    width = int(w//5)
    height = int(h//5)

    # canvas
    canvas = np.zeros([h, w, 3], dtype='uint8')
    canvas2 = np.zeros([h, w, 3], dtype='uint8')

    location1 = 0
    location2 = 0
    location3 = 0
    location4 = 0
    location5 = 0
    n_shades = 254

    getGradientPoint()

    for j in range(n_shades):
        canvas[0:height//2, location1:location1+w //
               n_shades] = (arr1_GBRPixel0[j], arr1_GBRPixel1[j], arr1_GBRPixel2[j])
        location1 += w//n_shades

    for j in range(n_shades):
        canvas[height//2:height*2//2, location2:location2+w //
               n_shades] = (arr2_GBRPixel0[j], arr2_GBRPixel1[j], arr2_GBRPixel2[j])
        location2 += w//n_shades

    for j in range(n_shades):
        canvas[height*2//2:height*3//2, location3:location3+w //
               n_shades] = (arr3_GBRPixel0[j], arr3_GBRPixel1[j], arr3_GBRPixel2[j])
        location3 += w//n_shades

    for j in range(n_shades):
        canvas[height*3//2:height*4//2, location4:location4+w //
               n_shades] = (arr4_GBRPixel0[j], arr4_GBRPixel1[j], arr4_GBRPixel2[j])
        location4 += w//n_shades

    for j in range(n_shades):
        canvas[height*4//2:height*5//2, location5:location5+w //
               n_shades] = (arr5_GBRPixel0[j], arr5_GBRPixel1[j], arr5_GBRPixel2[j])
        location5 += w//n_shades

    for i in range(5):
        cv.rectangle(canvas, (0, height*5//2+(50*i)), ((height*5//2) +
                     (height*5//2)+275, 300+(50*i)), GBRPixel[i], thickness=cv.FILLED)
        cv.putText(canvas, rgbToHex(int(GBRPixel[i][0]), int(GBRPixel[i][1]), int(
            GBRPixel[i][2])), (int(w//1.15), 25+(50*i)), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    clearGradient()  # clear list in gradient point

    main_img = cv.imread(img)

    if main_img.shape[1] >= 5000 or main_img.shape[0] >= 2500:
        main_img = cv.resize(main_img, (int(
            main_img.shape[1]*0.1), int(main_img.shape[0]*0.1)), cv.INTER_AREA)
    elif main_img.shape[1] >= 3000 or main_img.shape[0] >= 1750:
        main_img = cv.resize(main_img, (int(
            main_img.shape[1]*0.3), int(main_img.shape[0]*0.3)), cv.INTER_AREA)
    elif main_img.shape[1] >= 1920 or main_img.shape[0] >= 1080:
        main_img = cv.resize(main_img, (int(
            main_img.shape[1]*0.5), int(main_img.shape[0]*0.5)), cv.INTER_AREA)
    else:
        main_img = cv.resize(
            main_img, (int(main_img.shape[1]), int(main_img.shape[0])), cv.INTER_AREA)

    rgb = cv.cvtColor(main_img, cv.COLOR_BGR2RGB)
    rgb = rgb.reshape(main_img.shape[1]*main_img.shape[0], 3)
    clf = KMeans(n_clusters=5)
    color_label = clf.fit_predict(rgb)
    result = clf.cluster_centers_
    count = Counter(color_label)
    ordered_color = [result[i] for i in count.keys()]

    # reverse list RGB to BGR
    for x in range(5):
        temp = ordered_color[x][0]
        ordered_color[x][0] = ordered_color[x][2]
        ordered_color[x][2] = temp

    for x in range(5):
        for y in range(3):
            ordered_color[x][y] = math.floor(int(ordered_color[x][y]))

    for i in range(5):
        cv.rectangle(canvas2, (0, height*i), (w, w),
                     ordered_color[i], thickness=cv.FILLED)

    # cv.imwrite("saved.jpg", canvas)
    cv.imshow("Dominant Color", canvas2)
    cv.imshow("Linear Color", canvas)
    cv.waitKey(0)


if __name__ == "__main__":
    main(img)
