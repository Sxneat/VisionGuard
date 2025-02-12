
import cv2 as cv
import numpy as np
import random

black_piece = 0

result = {}

pages = ["./Data/page1.png","./Data/page2.png","./Data/page3.png","./Data/page4.png","./Data/page5.png","./Data/page6.png","./Data/TestCon.png"]

image = cv.imread("./Data/Testful.png")
imagecopy = image.copy()
imagecon = cv.imread("./Data/page1.png")

cv.imshow("Testful",image)
cv.imshow("Test",imagecon)

print(f"Image dimensions: {image.shape}")
print(f"Number of pixels: {image.size}")
print(f"Data type of image: {image.dtype}")

def select_page(page):
    imagecon = cv.imread(page)
    grey = cv.cvtColor(imagecon, cv.COLOR_BGR2GRAY)
    _, binary_thresh = cv.threshold(grey, 140, 255, cv.THRESH_BINARY)
    cv.imshow('black and white', binary_thresh)
    grey1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary_thresh2 = cv.threshold(grey1, 180, 255, cv.THRESH_BINARY)
    cv.imshow('black', binary_thresh2)
    contours, _ = cv.findContours(binary_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours
# print(f"Retrieved {_}")

def draw_contours(contours):
    cv.drawContours(image, contours, -1, (20, 130, 0), 2)
    cv.imshow('Contour Image', image)
    
def draw_rec(contours):
    x, y, w, h = cv.boundingRect(contours)
    return x,y,w,h

def check(binary_thresh2):
    is_black = False

    x, y, w, h = cv.boundingRect(contours)

    # print({binary_thresh2.shape})
    # brightness = cv.mean(binary_thresh2[y:y+h, x:x+w])
    # print (f"bright: {brightness}")

    if np.mean(image[y:y+h, x:x+w]) > 120:
        is_black = False
    else:
        is_black = True
    return is_black

draw_contours(contours)

# def main():
#     for i, page in enumerate(pages):
#         image = (draw_contours, page, draw_rec)
#         cv.imshow(f'Contour Image {i}', image)
#         # break

# cv.waitKey(0)
# cv.destroyAllWindows()    

# main()

page_check = 1

i = 0

number = 60
def sort(page_num, x):
    if page_num == 1:
        if 101 < x < 130:
            return "e"
        elif 76 < x <= 101:
            return "d"
        elif 51 < x <= 76:
            return "c"
        elif 24 < x <= 51:
            return "b"
        else:
            return "a"
    elif page_num in [2, 3]:
        if 262 < x < 300:
            return "e"
        elif 237 < x <= 262:
            return "d"
        elif 212 < x <= 237:
            return "c"
        elif 186 < x <= 212:
            return "b"
        else:
            return "a"
    elif page_num in [4, 5]:
        if 425 < x:
            return "e"
        elif 398 < x <= 425:
            return "d"
        elif 373 < x <= 398:
            return "c"
        elif 348 < x <= 373:
            return "b"
        else:
            return "a"


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

    # if check(binary_thresh2) == True:
    #     black_piece = black_piece + 1
    #     letter = sort(letter,number,x)
    #     # row = number//5 +1
    #     row = number
    #     if number % 5 == 0:
    #         row = row-1
    #     result[f"{row}"] = letter

def row(page_num, y, divider=29, divider_top=28):
    assert page_num in [1, 2, 3, 4, 5], "Invalid"
    divider = divider_top if page_num in [2, 4] else divider

    offset = 0 if mask_id in [2, 4] else 1
    row = (y - (divider * 10 * offset)) // divider
    return row + (1 - offset)

def main():
    for i, mask in enumerate(masks, start=1):
        drawed_img, contours = detect_contour_and_draw(src_image, mask, draw_as_rectangles=False)
        
        # for each contour, get the rectangle and classify the choice
        for j, contr in enumerate(contours):
            rect = cv.boundingRect(contr)
            x, y, w, h = rect
            choice = classify_choice(i, x)
            row = row(i, y)
                
            
            # Put Choice on the image at the center of the rectangle
            cv.putText(drawed_img, f"{row}{choice}", (x + (w//4)+1, y + (h//2)+1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv.imshow(f'Contour Image {i}', drawed_img)        
        # break

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


# print(":))")
# for x,y in result.items():
#     print(x,y)

print("Answer:", black_piece)
cv.waitKey(0)
cv.destroyAllWindows()