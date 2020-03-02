
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog
from PIL import *
from PIL import Image, ImageDraw, ImageDraw2, ImageFont
import numpy as np
import math


class everything:
    def binary_image(nrow,ncol,Value):
        x, y = np.indices((nrow, ncol))
        mask_lines = np.zeros(shape=(nrow,ncol))

        x0, y0, r0 = 30, 30, 10
        x1, y1, r1 = 70, 30, 10

        for i in range (50, 70):
            mask_lines[i][i] = 1
            mask_lines[i][i + 1] = 1
            mask_lines[i][i + 2] = 1
            mask_lines[i][i + 3] = 1
            mask_lines[i][i + 6] = 1
            mask_lines[i-20][90-i+1] = 1
            mask_lines[i-20][90-i+2] = 1
            mask_lines[i-20][90-i+3] = 1

        mask_square1 = np.fmax(np.absolute( x - x1), np.absolute( y - y1)) <= r1
        imge = np.logical_or(mask_lines, mask_square1) * Value

        return imge

    def np2PIL(im):
        print("size of arr: ",im.shape)
        img = Image.fromarray(im, 'RGB')
        return img

    def np2PIL_color(im):
        print("size of arr: ",im.shape)
        img = Image.fromarray(np.uint8(im))
        return img

    def threshold(im,T, LOW, HIGH):
        (nrows, ncols) = im.shape
        im_out = np.zeros(shape = im.shape)
        for i in range(nrows):
            for j in range(ncols):
                if abs(im[i][j]) <  T :
                    im_out[i][j] = LOW
                else:
                    im_out[i][j] = HIGH
        return im_out

    #Connected-component labeling
    def label_8_connected(bim, ONE):
        max_label = int(10000)
        nrow = bim.shape[0]
        ncol = bim.shape[1]
        print("nrow, ncol", nrow, ncol)
        im = np.zeros(shape=(nrow,ncol), dtype = int)
        a = np.zeros(shape=max_label, dtype = int)
        a = np.arange(0,max_label, dtype = int)
        color_map = np.zeros(shape = (max_label,3), dtype= np.uint8)
        color_im = np.zeros(shape = (nrow, ncol,3), dtype= np.uint8)

        for i in range(max_label):
            np.random.seed(i)

            color_map[i][0] = np.random.randint(0,255,1,dtype = np.uint8)
            color_map[i][1] = np.random.randint(0,255,1,dtype = np.uint8)
            color_map[i][2] = np.random.randint(0,255,1,dtype = np.uint8)

        k = 0
        for i in range(nrow):
            for j in range(ncol):
                im[i][j] = max_label
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                c   = bim[i][j]
                l   = bim[i][j - 1]
                u   = bim[i - 1 ][j]

                d   = bim[i - 1][j - 1]
                r   = bim[i - 1][j + 1]

                label_u  = im[i -1][j]
                label_l  = im[i][j - 1]

                label_d  = im[i - 1][j - 1]
                label_r  = im[i - 1][j + 1]

                im[i][j] = max_label
                if c == ONE:
                    min_label = min(label_u, label_l, label_d, label_r)
                    if min_label == max_label:
                        k += 1
                        im[i][j] = k
                    else:
                        im[i][j] = min_label
                        if min_label != label_u and label_u != max_label  :
                            everything.update_array(a, min_label, label_u)

                        if min_label != label_l and label_l != max_label  :
                            everything.update_array(a, min_label, label_l)

                        if min_label != label_d and label_d != max_label  :
                            everything.update_array(a, min_label, label_d)

                        if min_label != label_r and label_r != max_label  :
                            everything.update_array(a, min_label, label_r)

                else :
                    im[i][j] = max_label

        # final reduction in label array
        for i in range(k+1):
            index = i
            while a[index] != index:
                index = a[index]
            a[i] = a[index]

        #second pass to resolve labels and show label colors
        for i in range(nrow):
            for j in range(ncol):

                if bim[i][j] == ONE:
                    im[i][j] = a[im[i][j]]
                    if im[i][j] == max_label:
                        im[i][j] == 0

                        color_im[i][j][0] = 0
                        color_im[i][j][1] = 0
                        color_im[i][j][2] = 0

                    color_im[i][j][0] = color_map[im[i][j],0]
                    color_im[i][j][1] = color_map[im[i][j],1]
                    color_im[i][j][2] = color_map[im[i][j],2]


        return im

    #Finding the amount of labels (characters / numbers) in a given image
    def findLabelAmount(bim, im):
        nrow = bim.shape[0]
        ncol = bim.shape[1]

        list = []  #will be used to store labels

        #finding the number of labels
        counter = -1
        for i in range(nrow):
            for j in range(ncol):
                if list.__contains__(im[i][j]):
                    print(im[i][j])
                else:
                    list.append(im[i][j])
                    counter = counter + 1
        list.remove(10000)
        return counter, list

    #Drawing rectangles around the characters
    def drawRec(im, bim, counter, list, image):
        width, height = 4, counter
        rectangles = [[0 for x in range(width)] for y in range(height)] #will be used to store minx, miny, maxx, maxy values of the characters

        nrow = bim.shape[0]
        ncol = bim.shape[1]

        for i in range(nrow):
            for j in range(ncol):
                if list.__contains__(im[i][j]):
                    index = list.index(im[i][j])

                    if rectangles[index][0] == 0 and rectangles[index][1] == 0 and rectangles[index][2] == 0 and rectangles[index][3] == 0: #if this is the first time with the label
                        rectangles[index][0] = i
                        rectangles[index][1] = j
                        rectangles[index][2] = i
                        rectangles[index][3] = j
                    else:
                        if i < rectangles[index][0]:
                            rectangles[index][0] = i
                        if j < rectangles[index][1]:
                            rectangles[index][1] = j
                        if i > rectangles[index][2]:
                            rectangles[index][2] = i
                        if j > rectangles[index][3]:
                            rectangles[index][3] = j


        #DRAWING
        source_img = Image.open(image).convert("RGBA")
        draw = ImageDraw.Draw(source_img)

        for b in range(counter):
            draw.rectangle(((rectangles[b][1], rectangles[b][0]), (rectangles[b][3], rectangles[b][2])), fill=None,
                           outline='red', width=3)
            source_img.save("output.png", "PNG")

        return rectangles

    #Returns an array of moments of resized images -> all moments of the image
    def featureVectors(bim, rectangles, counter):

        arrayOfResizedImg = []  # will be used to store resized images

        #Resizing images
        for k in range(counter):

            #corners of the rectangle
            minx=rectangles[k][1]
            miny=rectangles[k][0]
            maxx=rectangles[k][3]
            maxy=rectangles[k][2]

            #Crop and store operations
            im1 = Image.fromarray(bim)
            im2 = im1.crop((minx,miny,maxx,maxy))
            im3 = im2.resize((21,21))
            arrayOfResizedImg.append(im3)

            featuresHu = [] #will be used to store the moments of resized images
            for i in range(len(arrayOfResizedImg)):
                featuresHu.append(everything.calcMomentsHu(arrayOfResizedImg[i]))

        return featuresHu

    def update_array(a, label1, label2) :
        index = lab_small = lab_large = 0
        if label1 < label2 :
            lab_small = label1
            lab_large = label2
        else :
            lab_small = label2
            lab_large = label1
        index = lab_large
        while index > 1 and a[index] != lab_small:
            if a[index] < lab_small:
                temp = index
                index = lab_small
                lab_small = a[temp]
            elif a[index] > lab_small:
                temp = a[index]
                a[index] = lab_small
                index = temp
            else: #a[index] == lab_small
                break
        return

    #Calculates the Hu Moment of an image
    def calcMomentsHu(image):
        f = np.asarray(image)
        nrow = f.shape[0]
        ncol = f.shape[1]

        rawMoments = [[0, 0], [0, 0]]

        for i in range(2):
            for j in range(2):
                for x in range(nrow):
                    for y in range(ncol):
                        rawMoments[i][j] += pow(x, i) * pow(y, j) * f[x][y]


        centralMoments = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        x_bar = rawMoments[1][0] / rawMoments[0][0]
        y_bar = rawMoments[0][1] / rawMoments[0][0]

        for i in range(4):
            for j in range(4):
                for x in range(nrow):
                    for y in range(ncol):
                        centralMoments[i][j] += pow(x - x_bar, i) * pow(y - y_bar, j) * f[x][y]


        scaleInvariants = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for i in range(4):
            for j in range(4):
                for x in range(nrow):
                    for y in range(ncol):
                        scaleInvariants[i][j] = centralMoments[i][j] / pow(centralMoments[0][0], (1 + ((i + j) / 2)))

        H1 = scaleInvariants[2][0] + scaleInvariants[0][2]
        H2 = pow((scaleInvariants[2][0] - scaleInvariants[0][2]), 2) + (4 * pow((scaleInvariants[1][1]), 2))
        H3 = pow((scaleInvariants[3][0] - (3 * scaleInvariants[1][2])), 2) + pow(((3 * scaleInvariants[2][1]) - scaleInvariants[0][3]), 2)
        H4 = pow((scaleInvariants[3][0] + scaleInvariants[1][2]), 2) + pow((scaleInvariants[2][1] + scaleInvariants[0][3]), 2)
        H5 = (scaleInvariants[3][0] - (3 * scaleInvariants[1][2])) * ((scaleInvariants[3][0] + scaleInvariants[1][2])) * \
             ((pow((scaleInvariants[3][0] + scaleInvariants[1][2]), 2)) - (3 * (pow((scaleInvariants[2][1] + scaleInvariants[0][3]), 2)))) + \
             ((3 * scaleInvariants[2][1]) - scaleInvariants[0][3]) * (scaleInvariants[2][1] + scaleInvariants[0][3]) * \
             ((3 * (pow((scaleInvariants[3][0] + scaleInvariants[1][2]), 2))) - pow((scaleInvariants[2][1] + scaleInvariants[0][3]), 2))
        H6 = (scaleInvariants[2][0] - scaleInvariants[0][2]) * (pow((scaleInvariants[3][0] + scaleInvariants[1][2]), 2) - pow((scaleInvariants[2][1] + scaleInvariants[0][3]), 2)) + (4 * scaleInvariants[1][1]) * (scaleInvariants[3][0] + scaleInvariants[1][2]) * (scaleInvariants[2][1] + scaleInvariants[0][3])

        H7 = (((3 * scaleInvariants[2][1]) - scaleInvariants[0][3]) * (scaleInvariants[3][0] + scaleInvariants[1][2]) * (pow((scaleInvariants[3][0] + scaleInvariants[1][2]), 2) - (3 * pow(scaleInvariants[2][1] + scaleInvariants[0][3], 2))) - \
              ((scaleInvariants[3][0] - (3 * scaleInvariants[1][2]))) * (scaleInvariants[2][1] + scaleInvariants[0][3]) * ((3 * pow((scaleInvariants[3][0] + scaleInvariants[1][2]), 2)) - ((pow((scaleInvariants[2][1] + scaleInvariants[0][3]), 2)))))


        #taking the log -> to be able to have a more correct match
        H1 = -1 * math.copysign(1.0, H1) * math.log10(abs(H1))
        H2 = -1 * math.copysign(1.0, H2) * math.log10(abs(H2))
        H3 = -1 * math.copysign(1.0, H3) * math.log10(abs(H3))
        H4 = -1 * math.copysign(1.0, H4) * math.log10(abs(H4))
        H5 = -1 * math.copysign(1.0, H5) * math.log10(abs(H5))
        H6 = -1 * math.copysign(1.0, H6) * math.log10(abs(H6))
        H7 = -1 * math.copysign(1.0, H7) * math.log10(abs(H7))

        return [H1,H2,H3,H4,H5,H6,H7]

    #Calculates the R Moment of an image
    def calcMomentsR(image):
        huMom = everything.calcMomentsHu(image)
        R1 = (pow(huMom[1], (1/2))) / (huMom[0])
        R2 = (huMom[0] + (pow(huMom[1], (1/2)))) / (huMom[0] - (pow(huMom[1], (1/2))))
        R3 = (pow(huMom[2], (1/2))) / (pow(huMom[3], (1/2)))
        R4 = (pow(huMom[2], (1/2))) / pow(abs(huMom[4]), (1/2))
        R5 = (pow(huMom[3], (1/2))) / pow(abs(huMom[4]), (1/2))
        R6 = abs(huMom[5]) / (huMom[0] * huMom[2])
        R7 = abs(huMom[5]) / (huMom[0] * (pow(abs(huMom[4]), (1/2))))
        R8 = abs(huMom[5]) / (huMom[2] * (pow(huMom[1], (1/2))))
        R9 = abs(huMom[5]) / pow((huMom[1] * abs(huMom[4])), (1/2))
        R10 = abs(huMom[4]) / (huMom[2] * huMom[3])

        return [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]

    #Finds the minimum value in a given array -> returns the index of that value
    def findMin(array):
        min = array[0]
        for i in range(len(array)):
            if array[i] < min:
                min = array[i]
        return (array.index(min))

    #Saves a given image as an array into a file
    def saveFile(image):
        moments = everything.findAllFeatureVectors(image) # calculating the moments of a given image
        fileName = image[0:-4] #determining the file name
        np.save(fileName, moments) #saving it as a file with the name of that image
        return fileName + ".npy" #returns the name of the file

    fileNames = [] #will store the files that are being used to train the program

    #Saves the moments of multiple images into files -> returns an array of file names
    def saveFiles(*args):
        for i in args:
            everything.fileNames.append(everything.saveFile(i))
        return everything.fileNames # returns an array of file names

    #Compares the moments of two images (in this case one of them is a traine image's moments and the other one is input image's moments
    def comparisonForOne(inputMom, trainMom):

        allDistances = [] #all character's distances to all train characters
        matches = [] #match values
        disFor1 = [] #one input character's distances to all train characters

        for i in range(len(inputMom)): #going through the input characters
            for j in range(len(trainMom)): #going through the train characters (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
                d1 = pow((inputMom[i][0] - trainMom[j][0]), 2)
                d2 = pow((inputMom[i][1] - trainMom[j][1]), 2)
                d3 = pow((inputMom[i][2] - trainMom[j][2]), 2)
                d4 = pow((inputMom[i][3] - trainMom[j][3]), 2)
                d5 = pow((inputMom[i][4] - trainMom[j][4]), 2)
                d6 = pow((inputMom[i][5] - trainMom[j][5]), 2)
                d7 = pow((inputMom[i][6] - trainMom[j][6]), 2)

                distance = pow((d1 + d2 + d3 + d4 + d5 + d6 + d7), (1/2)) # 1 input character's distance to 1 train character
                disFor1.append(distance)  # 1 input character's distances to all train characters

            allDistances.append(disFor1)
            disFor1 = []

        for m in range(len(allDistances)):
            matches.append(everything.findMin(allDistances[m]))

        return matches #returns the match values of all input characters

    #Finds & returns the most frequent number in a given list
    def findMostFrequent(list):
        counter = 0
        mostFreq = list[0]

        for i in list: #go through the given list
            current_frequency = list.count(i) #count the number of occurrences of each i in the list
            if(current_frequency > counter): # if it is more frequent than what we have
                counter = current_frequency # set it as the new 'most frequent'
                mostFreq = i

        return mostFreq

    #Compares the characters of a given image with all of the train images
    def comparisonForAll(image, fileNames): #takes an image to process and an array of files with train image's moments
        allMatches = []
        inputMom = everything.findAllFeatureVectors(image)
        finalMatches = [] #matched characters for all characters of the given image

        for i in fileNames: #for every train image
            realMom = np.load(i) #loading the train moments into array
            matchesForOne = everything.comparisonForOne(inputMom, realMom) #all character's match to one font (one train image)
            allMatches.append(matchesForOne) #collecting all the matches

        transposedNP = np.asarray(allMatches).transpose() #taking the transposed version of allMatches array to itare it properly
        transposed = transposedNP.tolist() #taking np array as a list

        for j in range(len(transposed)):
            finalMatches.append(everything.findMostFrequent(transposed[j]))

        return finalMatches

    #Calculates the feature vectors of an image an returns it as a 2d array
    def findAllFeatureVectors(input_img):
        img = Image.open(input_img)  # image to process

        img_gray = img.convert('L')  # converts the image to grayscale image
        ONE = 150
        a = np.asarray(img_gray)  # from PIL to np array
        a_bin = everything.threshold(a, 150, ONE, 0)
        im = Image.fromarray(a_bin)  # from np array to PIL format
        a_bin = np.asarray(im)

        image_name = input_img
        imO = everything.label_8_connected(a_bin, ONE)
        counterO, listO = everything.findLabelAmount(a_bin, imO)
        rectanglesO = everything.drawRec(imO, a_bin, counterO, listO, image_name)
        featuresHuO = everything.featureVectors(a_bin, rectanglesO, counterO)

        return featuresHuO

    #Writing the final match values on image
    #takes an image to write on, takes what to write, takes xs and ys
    def addText(image, finalMatches, rectangles):
        source_img = Image.open(image).convert("RGBA")
        draw = ImageDraw.Draw(source_img)
        fnt = ImageFont.truetype('/Library/Fonts/Helvetica.ttc', 20)

        #go through final matches
        for j in range(len(finalMatches)):
            minx = rectangles[j][1]
            miny = rectangles[j][0]
            draw.text((minx + 30 ,miny - 30), str(finalMatches[j]), font=fnt, fill='red')
        source_img.save("output1.png", "PNG")
        source_img.show()

        # Matches the numbers in the given image
    def matchNumbers(input_img, fileNames):
        img = Image.open(input_img)  # image to process

        img_gray = img.convert('L')  # converts the image to grayscale image
        ONE = 150
        a = np.asarray(img_gray)  # from PIL to np array
        a_bin = everything.threshold(a, 150, ONE, 0)
        im = Image.fromarray(a_bin)  # from np array to PIL format
        a_bin = np.asarray(im)

        image_name = input_img
        imO = everything.label_8_connected(a_bin, ONE)
        counterO, listO = everything.findLabelAmount(a_bin, imO)
        rectanglesO = everything.drawRec(imO, a_bin, counterO, listO, image_name)
        featuresHuO = everything.featureVectors(a_bin, rectanglesO, counterO)
        finalMatchesO = everything.comparisonForAll(input_img, fileNames)
        everything.addText('output.png', finalMatchesO, rectanglesO)

##################################### USER INTERFACE #####################################

root = tk.Tk() #creates window
warningLabel2 = tk.Label(root, text="This program can be used to define numbers in a given image.")
warningLabel2.pack()
warningLabel = tk.Label(root, text="When you are training the program, please make sure that you are training the program with the right kind of font.\nAdding fonts that are too diversed might lower the correctness of the program.")
warningLabel.pack()

#Adds a train image to use for comparison
def TrainClicked(event):

# selecting an image
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                            filetypes=(("png files", "*.png"), ("all files", "*.*")))
    img = root.filename
    everything.saveFiles(img)

    messagebox.showinfo(" ", "Train image added.")
    print(everything.fileNames)

#Finds the numbers in a given image
def FindNumbersClicked(event):

    # selecting image
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                            filetypes=(("png files", "*.png"), ("all files", "*.*")))
    img = root.filename

    everything.matchNumbers(img, everything.fileNames)

#BUTTONS
TrainButton = tk.Button(root, text ="Train", highlightbackground='#3E4149')
TrainButton.bind("<Button-1>", TrainClicked)
TrainButton.pack()

FindNumbersButton = tk.Button(root, text="Load an Image", highlightbackground='#3E4149')
FindNumbersButton.bind("<Button-1>", FindNumbersClicked)
FindNumbersButton.pack()

root.mainloop()
