import os, sys
import PIL
from PIL import Image
#from resizeimage import resizeimage

def checkCorrupt(imageName):
    try:
        image = imageName
        img = Image.open(image)
        img.verify()
        img.close()
    except:
        print("corrupt " + str(imageName))
        #os.remove(imageName)

def main():
    # Open a file
    dirName = "../data/downloads/"

    os.chdir(dirName)
    dirs = os.listdir(os.getcwd())

    for file in dirs:
        if(file != ".DS_Store"): 
            os.chdir(file)
            allPictures = os.listdir(os.getcwd())
            print(file)
            for image in allPictures:
                #print(image)
                checkCorrupt(image)
            allPictures = os.listdir(os.getcwd())
            count = 1
            for image in allPictures:
                #os.rename(image, str(count) + ".jpg")
                count = count + 1 
            #os.chdir("..")
        else:
            os.remove(".DS_Store")
    print("Total fine images: ", count-1)
main()
