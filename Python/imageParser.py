from os import listdir
from image import Image

class ImageParser:
    featureVectorDict = dict() # Relates file name to file properties

    def __init__(self):
        self.imageFileNames = self.loadImages()
        self.setAllFeatureVectors()

    def loadImages(self) -> list:
        ## Get all the files in the ImageFIles folder matching png, jpg, etc. 
        return ['Hello World']


    # Iterates over the image files to extract out the feature vectors
    def setAllFeatureVectors(self):
        for imageName in self.imageFileNames:
            image = Image()
            image.fileName = imageName
            image.label = 0 # TODO - Figure out how/where we decide the label
            image.featureVectors = self.parseFeatureVectors(imageName)
            self.featureVectorDict[imageName] = image

    def parseFeatureVectors(self, imageName):
        ## Put code here to store the feature vector as a list
        return [1, 2, 3]
        
