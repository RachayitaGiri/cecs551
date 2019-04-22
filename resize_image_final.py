import glob
import os, sys
import PIL
from PIL import Image
import progressbar as pb

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
        global fineCount
        fineCount = fineCount + 1
        #os.rename(imageName, str(newName) + ".jpeg")
    except:
        global corruptCount
        print("corrupt " + str(imageName))
        corruptCount = corruptCount + 1

#folderName = input("Please enter the directory name: ")
#folderName = "../datasets" + sys.argv[1]

# progress bar variables
progress = pb.ProgressBar(widgets="#",maxval=500000).start()
progvar = 0

allPictures = glob.glob('/home/datasets/%s/*.jpg' % sys.argv[1])
totalCount = 0
fineCount = 0
corruptCount = 0
for image in allPictures:
    #print(image)
    resize(image)
    progress.update(progvar + 1)
    progvar += 1
    totalCount = totalCount + 1
print("Total images scanned: ", totalCount)
print("Fine images resized: ", fineCount)
print("Corrupt images: ", corruptCount)
#def main(): 

#main()
