from PIL import *
from PIL import Image
import numpy as np

def main():
    #img = readPILimg()
    #arr = PIL2np(img)
    ONE = 150  # 1-valued pixel intensity
    bim = binary_image(100,100,ONE)
    new_img = np2PIL(bim)
    new_img.show()
    #label = blob_coloring_4_connected(bim, ONE)
    labels = levialdi_blob(bim, ONE)  # you should implement it
    #print("number of object = ",num_object)
    new_img2 = np2PIL(labels)
    new_img2.show()



def binary_image(nrow,ncol,Value):
#creates a binary image with size nrow x ncol with pixel values Value
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

    #mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute( x - x1), np.absolute( y - y1)) <= r1
    #mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    #mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    #mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    #imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value

    return imge

def readPILimg(): #read image file using PIL
    img = Image.open('/Users/gokmen/Dropbox/vision-python/images/brick-house.png')
    img.show()
    img_gray = color2gray(img)
    img_gray.show()
    #img_gray.save('/Users/gokmen/Dropbox/vision-python/images/brick-house-gs','png')
    #new_img = img.resize((256,256))
    #new_img.show()
    return img_gray

def color2gray(img):
# converts color image to gray level image (L: gray level, 1:binary image)
    img_gray = img.convert('L')
    return img_gray

def PIL2np(img): #converts image in PIL format to a numpy array
    nrows = img.size[0]
    ncols = img.size[1]
    print("nrows, ncols : ", nrows,ncols)
    imgarray = np.array(img.convert("L"))
    return imgarray

def np2PIL(im): #converts numpy array to a PIL image
    print("size of arr: ",im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


if __name__=='__main__':
    main()
