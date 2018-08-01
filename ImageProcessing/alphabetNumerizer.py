import cv2
import numpy as np
import pymysql.cursors

directories = ["Test Set", "Alphabet", "letters", "lettersR"]
fileType = ["jpg", "png", "png", "jpg"]
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
for x in range(0,4):
    for i in range(0,len(letters)):
        classification = letters[i]
        # Skew Coreection
        imgDir = 'C:\\Users\\Ethan Ray Mosqueda\\PycharmProjects\\EMain\\ImageProcessing\\imgs\\' + directories[
            x] + '\\' + letters[i] + '.' + fileType[x]
        # imgDir = 'C:\\Users\\Ethan Ray Mosqueda\\PycharmProjects\\EMain\\ImageProcessing\\imgs\\Skewed\\A.jpg'
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

        # cv2.putText(image, "Angle: {:.2f} degrees".format(angle),
        # 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # print("[INFO] angle: {:.3f}".format(angle))
        # cv2.imshow("Rotated", rotated)
        # cv2.imshow("Input", input)
        # cv2.waitKey(0)

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
                if image[row][col] >= 40:
                    image[row][col] = 255
                else:
                    image[row][col] = 0
        # cv2.imshow("binarized",image)
        # cv2.waitKey(0)
        # Getting the Bounds
        upperBounds, lowerBounds, leftBounds, rightBounds = -1, -1, -1, -1
        # upperBounds
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
        for row in range(height - 1, 0, -1):
            for col in range(width - 1, 0, -1):
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
        for col in range(width - 1, 0, -1):
            for row in range(height - 1, 0, -1):
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
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        resized_image = cv2.resize(crop_img, (400, 400))
        # cv2.imshow("resized", resized_image)
        # cv2.waitKey(0)

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
            elif row < rowBlockSize * 2:
                h2 += rowAvg
            elif row < rowBlockSize * 3:
                h3 += rowAvg
            else:
                h4 += rowAvg

        for col in range(colSize):
            for row in range(rowSize):
                colAvg = colAvg + resized_image[row][col]

            colAvg = colAvg / colSize

            if col < colBlockSize:
                w1 += colAvg
            elif col < colBlockSize * 2:
                w2 += colAvg
            elif col < colBlockSize * 3:
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

        lets = {
            "A":0 , "B":1 , "C":2 , "D":3 ,
            "E":4 , "F":5 , "G":6 , "H":7 ,
            "I":8 , "J":9 , "K":10, "L":11,
            "M":12, "N":13, "O":14, "P":15,
            "Q":16, "R":17, "S":18, "T":19,
            "U":20, "V":21, "W":22, "X":23,
            "Y":24, "Z":25
        }
        print(lets[classification])

        # Connect to the database
        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='',
                                     db='alphabet',
                                     charset='utf8mb4',
                                     cursorclass=pymysql.cursors.DictCursor)

        try:
            with connection.cursor() as cursor:
                # Create a new record
                sql = "INSERT INTO `img` (`h1`, `h2`, `h3`, `h4`, `w1`, `w2`, `w3`, `w4`, `class`) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (float(h1), float(h2), float(h3), float(h4), float(w1), float(w2), float(w3), float(w4), lets[classification]))

            # connection is not autocommit by default. So you must commit to save
            # your changes.
            connection.commit()

        finally:
            connection.close()
