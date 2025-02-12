import cv2 as cv
import numpy as np
import random

blue_piece = 0
green_piece = 0

image = cv.imread("./Data/Checker.png")
imagecopy = image.copy()

cv.imshow("Checker",image)

print(f"Image dimensions: {image.shape}")
print(f"Number of pixels: {image.size}")
print(f"Data type of image: {image.dtype}")

grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

_, binary_thresh = cv.threshold(grey, 125, 255, cv.THRESH_BINARY)
cv.imshow('black and white', binary_thresh)

contours, _ = cv.findContours(binary_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print(f"Retrieved {_}")
cv.drawContours(image, contours, -1, (0, 130, 200), 2)
cv.imshow('Contour Image', image)
    

def check(image):

    # print("Cv")
    # print(cv.mean(image))
    print({image.shape})
    blue, green, red, _1 = cv.mean(imagecopy[y:y+h, x:x+w])

    print (f"blue: {blue}, Green: {green}")

    is_green = True if green > blue else False
    return is_green

i = 0

for contr in contours:
    rect = cv.boundingRect(contr)
    x, y, w, h = rect

    is_green = check(imagecopy[y:y+h, x:x+w])

    if w < 21:
        continue

    if h < 22:
        continue

    if y < 150 and y > 40:
        continue

    if np.mean(image[y:y+h, x:x+w]) > 120:
        continue
    
    print("mean")
    print(is_green)

    i += 1
    print (f"num-{i}")

    # is_green = check(image[y:y+h, x:x+w])

    if is_green is True:
        green_piece += 1
        cv.rectangle(
            imagecopy,
            (x, y),
            (x + w, y + h),
            (200, 210, 0),
            1
        )   

    else:
        blue_piece += 1
        cv.rectangle(
            imagecopy,
            (x, y),
            (x + w, y + h),
            (0, 210, 250),
            1
        )   



    cv.putText(imagecopy, f"{i}", (x+2, y+7), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    cv.imshow('Drawing Functions', imagecopy)
    
cv.imshow('Check img', imagecopy)


cv.imshow('Bounding Box', imagecopy)

print("Blue pieces:", blue_piece)
print("Green pieces:", green_piece)
cv.waitKey(0)
cv.destroyAllWindows()



# blur_kernel = np.array([[0.2,0.2,0.1],[0,0,0],[0.2,0.2,0.1]])
# blur = cv.filter2D(image, -1,blur_kernel)
# cv.imshow('Blurred', blur)
# cv.waitKey(0)


# edges = cv2.Canny(image, 100, 200)
# cv.imshow('Edges', edges)
# image = img_temp.copy()  

# Drawing functions
# cv.line(image, (0, 0), (150, 150), (255, 0, 0), 5)
# cv.rectangle(image, (50, 50), (100, 100), (0, 255, 0), 3)
# cv2.circle(image, (120, 120), 30, (0, 0, 255), -1)
# cv2.putText(image, 'OpenCV', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# cv2.imshow('Drawing Functions', image)
# image = img_temp.copy()  # Reset the image to original

# # Contour detection
# contours, _ = cv2.findContours(binary_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
# cv2.imshow('Contour Image', image)
# image = img_temp.copy()  # Reset the image to original
# # Display the results of all operations
# cv2.waitKey(0)
# cv2.destroyAllWindows()