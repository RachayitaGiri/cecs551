import os, sys
import PIL
from PIL import Image
#from resizeimage import resizeimage

def resize(imageName):
    try:
        basewidth = 224
        image = imageName
        img = Image.open(image)
        img.verify()
        img.close()

        img = Image.open(image)
        img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)
        img.save(imageName, quality=90)
        img.close()
        #os.rename(imageName, str(newName) + ".jpeg") 
    except:
        print("corrupt " + str(imageName))
        #os.remove(imageName)

def main():
    # Open a file
    dirName = "valtest"

    os.chdir(dirName)
    dirs = os.listdir(os.getcwd())

    for file in dirs:
        if(file != ".DS_Store"): 
            #os.chdir(file)
            allPictures = os.listdir(os.getcwd())
            print(file)
            for image in allPictures:
                #print(image)
                resize(image)
            allPictures = os.listdir(os.getcwd())
            count = 1
            for image in allPictures:
                #os.rename(image, str(count) + ".jpg")
                count = count + 1 
            #os.chdir("..")
        else:
            os.remove(".DS_Store")
    print("Total images resized: ", count-1)
main()
