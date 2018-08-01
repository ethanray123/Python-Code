import cv2
import numpy as np
import pandas as pd
from svmutil import *
from svm import *
import pymysql.cursors
import matplotlib.pyplot as plt

checks = 0
iterations = 0
# Training directories with their file types
# directories = ["Test Set", "Alphabet", "letters", "lettersR"]
# fileType = ["jpg", "png", "png", "jpg"]

# Test directory with its file type
directories = ["lettersJa", "Skewed"]
fileType = ["jpg", "jpg"]
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
for x in range(0,1): #len(directories)
    for i in range(0,1): #len(letters)
        # Skew Coreection
        # imgDir = 'C:\\Users\\Ethan Ray Mosqueda\\PycharmProjects\\EMain\\ImageProcessing\\imgs\\'+directories[x]+'\\'+letters[i]+'.'+fileType[x]
        # imgDir = 'C:\\Users\\Ethan Ray Mosqueda\\PycharmProjects\\EMain\\ImageProcessing\\imgs\\Skewed\\A.jpg'
        imgDir = 'C:\\Users\\Ethan Ray Mosqueda\\Pictures\\Camera Roll\\Test5.jpg'
        input = cv2.imread(imgDir)
        gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)

        thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # cv2.imshow("thresh",thresh)
        # cv2.waitKey()
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)

        else:
            angle = -angle

        # rotate the image to deskew it
        (h, w) = input.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(input, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
        	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # print("[INFO] angle: {:.3f}".format(angle))
        cv2.imshow("Rotated", rotated)
        cv2.imshow("Input", input)
        cv2.waitKey(0)

        # Grayscaling
        image = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("grayed", image)
        # cv2.waitKey(0)
        # cv2.IMREAD_GRAYSCALE
        # IMREAD_GRAYSCALE = 0
        # IMREAD_COLOR = 1
        # IMREAD_UNCHANGED = -1

        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Getting Shape
        height, width = image.shape
        # print("height: "+str(height))
        # print("width: "+str(width))

        # Binarize
        for row in range(0, height):
            for col in range(0, width):
                if image[row][col] >= 65:
                    image[row][col] = 255
                else:
                    image[row][col] = 0
        # cv2.imshow("binarized",image)
        # cv2.waitKey(0)
        # Getting the Bounds
        upperBounds, lowerBounds, leftBounds, rightBounds = -1, -1, -1, -1
        #upperBounds
        dCol = -1
        for row in range(0, height):
            for col in range(0, width):
                if image[row][col] == 0:
                    upperBounds = row
                    dCol = col
                    # print("\nub init")
                    break
            if image[row][dCol] == 0:
                break

        # lowerBounds
        dCol = -1
        for row in range(height-1,0,-1):
            for col in range(width-1,0,-1):
                if image[row][col] == 0:
                    lowerBounds = row
                    dCol = col
                    # print("lb init")
                    break
            if image[row][dCol] == 0:
                break

        # leftBounds
        dRow = -1
        for col in range(0, width):
            for row in range(0, height):
                if image[row][col] == 0:
                    leftBounds = col
                    dRow = row
                    # print("lftb init")
                    break
            if image[dRow][col] == 0:
                break

        # rightBounds
        dRow = -1
        for col in range(width-1,0,-1):
            for row in range(height-1,0,-1):
                if image[row][col] == 0:
                    rightBounds = col
                    dRow = row
                    # print("rgtb init")
                    break
            if image[dRow][col] == 0:
                break

        while lowerBounds < height and lowerBounds % 4 != 0:
            lowerBounds = lowerBounds + 1

        while upperBounds > 0 and upperBounds % 4 != 0:
            upperBounds = upperBounds - 1

        while leftBounds > 0 and leftBounds % 4 != 0:
            leftBounds = leftBounds - 1

        while rightBounds < width and rightBounds % 4 != 0:
            rightBounds = rightBounds + 1


        #checking the bounds
        # print("\nlowerBounds: "+str(lowerBounds))
        # print("upperBounds: "+str(upperBounds))
        # print("leftBounds: "+str(leftBounds))
        # print("rightBounds: "+str(rightBounds))
        # print("\n\n")
        crop_img = image[upperBounds:lowerBounds, leftBounds:rightBounds]
        cv2.imshow("cropped", crop_img)
        resized_image = cv2.resize(crop_img, (400, 400))
        cv2.imshow("resized", resized_image)
        cv2.waitKey(0)

        rowSize = 400
        colSize = 400
        rowBlockSize = rowSize / 4
        colBlockSize = colSize / 4
        rowAvg, h1, h2, h3, h4, = 0, 0, 0, 0, 0
        colAvg, w1, w2, w3, w4 = 0, 0, 0, 0, 0

        for row in range(rowSize):
            for col in range(colSize):
                rowAvg = rowAvg + resized_image[row][col]
            rowAvg = rowAvg / rowSize

            if row < rowBlockSize:
                h1 += rowAvg
            elif row < rowBlockSize*2:
                h2 += rowAvg
            elif row < rowBlockSize*3:
                h3 += rowAvg
            else:
                h4 += rowAvg

        for col in range(colSize):
            for row in range(rowSize):
                colAvg = colAvg + resized_image[row][col]

            colAvg = colAvg / colSize

            if col < colBlockSize:
                w1 += colAvg
            elif col < colBlockSize*2:
                w2 += colAvg
            elif col < colBlockSize*3:
                w3 += colAvg
            else:
                w4 += colAvg


        h1 = h1 / rowBlockSize
        h2 = h2 / rowBlockSize
        h3 = h3 / rowBlockSize
        h4 = h4 / rowBlockSize

        w1 = w1 / colBlockSize
        w2 = w2 / colBlockSize
        w3 = w3 / colBlockSize
        w4 = w4 / colBlockSize

        # print("\nh1: "+str(h1))
        # print("h2: "+str(h2))
        # print("h3: "+str(h3))
        # print("h4: "+str(h4))
        # print("\nw1: "+str(w1))
        # print("w2: "+str(w2))
        # print("w3: "+str(w3))
        # print("w4: "+str(w4))

        # Capture DB data from csv

        #------------------Testing the SVM-------------------------------------------
        df = pd.read_csv('alphaData.csv')
        df.isnull().any()
        df = df.fillna(method='ffill') #for NaN vals in data coz the program said there was
        y = np.array(df['class'])
        X = np.array(df.drop(['class'], 1)) #, 1

        # print(X)
        # print(y)
        prob  = svm_problem(y, X.tolist())
        param = svm_parameter('-t 0 -c 4 -b 1')
        m = svm_train(prob, param)


        x0, max_idx =  gen_svm_nodearray([h1 ,h2 , h3, h4, w1, w2, w3, w4])
        label = libsvm.svm_predict(m, x0)

        lets = {
            0: "A", 1: "B", 2: "C", 3: "D",
            4: "E", 5: "F", 6: "G", 7: "H",
            8: "I", 9: "J", 10: "K", 11: "L",
            12: "M", 13: "N", 14: "O", 15: "P",
            16: "Q", 17: "R", 18: "S", 19: "T",
            20: "U", 21: "V", 22: "W", 23: "X",
            24: "Y", 25: "Z"
        }

        # print(x0)
        print("Test"+str(i))
        print("Prediction: "+lets[label])
        print("Actual: "+letters[i])
        if(lets[label] == letters[i]):
            checks = checks + 1
        iterations = iterations + 1


print("Score : ("+str(checks)+"/"+str(iterations)+")")
# ------------------Predicting the Class--------------------------------------
# example_measures = np.array([[h1,h2,h3,h4,w1,w2,w3,w4]])
# example_measures = example_measures.reshape(len(example_measures), -1) #
# prediction = clf.predict(example_measures)
# lets = {
#          0:"A",  1:"B",  2:"C",  3:"D",
#          4:"E",  5:"F",  6:"G",  7:"H",
#          8:"I",  9:"J", 10:"K", 11:"L",
#         12:"M", 13:"N", 14:"O", 15:"P",
#         16:"Q", 17:"R", 18:"S", 19:"T",
#         20:"U", 21:"V", 22:"W", 23:"X",
#         24:"Y", 25:"Z"
#         }
#print(prediction)

    #
    # # Connect to the database
    # connection = pymysql.connect(host='localhost',
    #                              user='root',
    #                              password='',
    #                              db='alphabet',
    #                              charset='utf8mb4',
    #                              cursorclass=pymysql.cursors.DictCursor)
    #
    # try:
    #     with connection.cursor() as cursor:
    #         # Create a new record
    #         sql = "INSERT INTO `img` (`h1`, `h2`, `h3`, `h4`, `w1`, `w2`, `w3`, `w4`, `class`) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    #         cursor.execute(sql, (float(h1), float(h2), float(h3), float(h4), float(w1), float(w2), float(w3), float(w4), classification))
    #
    #     # connection is not autocommit by default. So you must commit to save
    #     # your changes.
    #     connection.commit()
    #
    #     with connection.cursor() as cursor:
    #         # Read a single record
    #         sql = "SELECT `h1`, `h2`, `h3`, `h4`, `w1`, `w2`, `w3`, `w4` FROM `img` WHERE `class`=%s"
    #         cursor.execute(sql, 'A')
    #         result = cursor.fetchone()
    #         print(result)
    # finally:
    #     connection.close()

#
# for x in range(0,10):
#     for y in range(0,10):
#         print(img[x][y])
# cv2.imwrite('C:\\Users\\Ethan Ray Mosqueda\\PycharmProjects\\EMain\\ImageProcessing\\imgs\\rA.png', image)
