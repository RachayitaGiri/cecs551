# file to obtain a small subset of the original dataset for local testing
import os
import shutil

trainDir = os.listdir("../train2014/")
testDir = os.listdir("../test2014/")
valDir = os.listdir("../val2014/")

print("Train length: ", len(trainDir))
print("Test length: ", len(testDir))
print("Val length: ", len(valDir))

trainSet = trainDir[:100]
testSet = testDir[:50]
valSet = valDir[:50]

for img in trainSet:
    print(img)
    shutil.copyfile("../train2014/"+img, "train/"+img)
print("Extracted training subset.")
 
for img in testSet:
    print(img)
    shutil.copyfile("../test2014/"+img, "test/"+img)
print("Extracted testing subset.")

for img in valSet:
    print(img)
    shutil.copyfile("../val2014/"+img, "val/"+img)
print("Extracted validation subset.")

