
import cv2 as cv
import numpy as np
import random

black_piece = 0

result = {}

image = cv.imread("./Data/Testful.png")
imagecopy = image.copy()
imagecon = cv.imread("./Data/TestCon.png")
cv.imshow("Testful",image)

print(f"Image dimensions: {image.shape}")
print(f"Number of pixels: {image.size}")
print(f"Data type of image: {image.dtype}")

grey = cv.cvtColor(imagecon, cv.COLOR_BGR2GRAY)

_, binary_thresh = cv.threshold(grey, 140, 255, cv.THRESH_BINARY)
cv.imshow('black and white', binary_thresh)

grey1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

_, binary_thresh2 = cv.threshold(grey1, 140, 255, cv.THRESH_BINARY)
cv.imshow('black', binary_thresh2)

contours, _ = cv.findContours(binary_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print(f"Retrieved {_}")
cv.drawContours(image, contours, -1, (20, 130, 0), 2)
cv.imshow('Contour Image', image)
    
def check(binary_thresh2):
    is_black = False

    # print({binary_thresh2.shape})
    brightness = cv.mean(binary_thresh2[y:y+h, x:x+w])

    # print (f"bright: {brightness}")

    if np.mean(image[y:y+h, x:x+w]) > 120:
        is_black = False
    else:
        is_black = True
    return is_black

i = 0

number = 60

def sort(letter,number,x):

    if (number + 2) % 3 == 0:
        if x > 101 and x < 130:
            letter["e"] = 1
        if x < 101 and x > 76:
            letter["d"] = 1
        if x < 76 and x > 51:
            letter["c"] = 1
        if x < 51 and x > 24:
            letter["b"] = 1
        if x < 24 :
            letter["a"] = 1

    if (number + 3) % 3 == 0:
        if x > 262 and x < 300:
            letter["e"] = 1
        if x < 262 and x > 237:
            letter["d"] = 1
        if x < 237 and x > 212:
            letter["c"] = 1
        if x < 212 and x > 186:
            letter["b"] = 1
        if x < 186 and x > 130:
            letter["a"] = 1

    if number % 4 == 0:
        if x > 425:
            letter["e"] = 1
        if x < 425 and x > 398:
            letter["d"] = 1
        if x < 398 and x > 373:
            letter["c"] = 1
        if x < 373 and x > 348:
            letter["b"] = 1
        if x < 348 and x > 300:
            letter["a"] = 1
    # print(f"sort {number+1},{letter}")
    return letter


for contr in contours:
    rect = cv.boundingRect(contr)
    x, y, w, h = rect 

    letter = { "a" : 0, "b" : 0, "c" :0 , "d" : 0, "e" : 0}


    # is_green = check(imagecopy[y:y+h, x:x+w])

    # if np.mean(image[y:y+h, x:x+w]) > 120:
    #     continue
    if w < 21:
        continue

    if h < 22:
        continue

    i += 1
    # print (f"num-{i}")

    # print(f"pos {number}")

    # print(check(binary_thresh2))

    if check(binary_thresh2) == True:
        black_piece = black_piece + 1
        letter = sort(letter,number,x)
        # row = number//5 +1
        row = number
        if number % 5 == 0:
            row = row-1
        result[f"{row}"] = letter

    if (i-1) % 5 == 0:
        cv.putText(image, f"{number}", (x+25, y+24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.imshow('Position list', image)
        if number > 0:
            number = number - 1

    cv.rectangle (
        image,
        (x , y ),
        (x + w, y + h),
        (100, 230, 150),
        2,
        
    )   

    cv.putText(imagecopy, f"{i}", (x+2, y+7), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    cv.imshow('Drawing Functions', imagecopy)
    
cv.imshow('Bounding Box', image)


print(":))")
for x,y in result.items():
    print(x,y)

print("Answer:", black_piece)
cv.waitKey(0)
cv.destroyAllWindows()