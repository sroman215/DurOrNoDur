from os import listdir, getcwd
from os.path import isfile, join
from image import Image

class ImageParser:
    featureVectorDict = dict() # Relates file name to file properties

    # Constructor; hit on initialization
    def __init__(self):
        self.imageFileNames = self.loadImages()
        self.setAllFeatureVectors()

    # Load all images in our images directory for later use
    def loadImages(self) -> list:
        # Generate the string for the ImageFiles directory
        imageFileDir = self.getImageFilesDir()

        # Get only the files from the directory
        onlyFiles = self.getFilesInDir(imageFileDir)

        ## Only return files in the ImageFiles folder matching png, jpg, etc. 
        return self.filterFiles(onlyFiles)

    # Define the allowed file types, then filter using an insanely convoluted 1 liner. Man I hate Python
    def filterFiles(self, files): 
        allowedFileTypes = ['png', 'jpg', 'gif', 'bmp', 'svg']
        return [k for k in files if any(allowedType in k for allowedType in allowedFileTypes) ]

    # A simple 1 liner to extract only the files out of the chosen directory
    def getFilesInDir(self, directory): 
        return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

    # Python is dumb and doesn't understand ../ so I made something more complicated to arbitrarily get the correct directory path
    def getImageFilesDir(self):
        # Declare constants
        rootDirName = 'DurOrNoDur'
        currentFileDir = getcwd()

        # Format the directory structure to point to ImageFiles regardness of format/start location
        sliceIndex = currentFileDir.index(rootDirName)
        rootDir = currentFileDir[0 : sliceIndex + len(rootDirName) + 1]
        rootDir = rootDir[0:len(rootDir)-1] if rootDir[-1] == "\\" else  rootDir 
        return f"{rootDir}\ImageFiles"

    # Iterates over the image files to extract out the feature vectors
    def setAllFeatureVectors(self):
        for imageName in self.imageFileNames:
            featureVectors = self.parseFeatureVectors(imageName)
            label = 0 # TODO - Think of a way to do the label setting
            self.featureVectorDict[imageName] = Image(imageName, label, featureVectors)

    # Extract out the feature vectors for a given image file
    def parseFeatureVectors(self, imageName):
        ## Put code here to store the feature vector as a list
        return [1, 2, 3]
        
