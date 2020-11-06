from os import listdir, getcwd
from os.path import isfile, join
from image import Image

class ImageParser:
    imageDict = dict() # Relates file name to file properties
    durImages = [] # Initialize list for the image files that have a dur in them
    noDurImages = [] # Initialize list for hte image files that DO NOT have a dur in them

    # Constructor; hit on initialization
    def __init__(self) -> None:
        self.loadImages()
        self.setAllFeatureVectors(self.durImages, 1)
        self.setAllFeatureVectors(self.noDurImages, 0)

    # Load all images in our images directory for later use
    def loadImages(self) -> None:
        # Generate the string for the ImageFiles directory
        imageFileDir = self.getImageFilesDir()

        # Get only the files from the directory
        durImagesRaw = self.getFilesInDir(f"{imageFileDir}\Dur")
        noDurImagesRaw = self.getFilesInDir(f"{imageFileDir}\\NoDur") # \N represents new line, so we need an escape character
        
        ## Only return files in the ImageFiles folder matching png, jpg, etc. 
        self.durImages = self.filterFiles(durImagesRaw)
        self.noDurImages = self.filterFiles(noDurImagesRaw)

    def printImageDictValues(self) -> None: 
        for image in list(self.imageDict.values()):
            image.printValues()

    # Define the allowed file types, then filter using an insanely convoluted 1 liner. Man I hate Python
    def filterFiles(self, files)-> list: 
        allowedFileTypes = ['png', 'jpg', 'gif', 'bmp', 'svg']
        return [k for k in files if any(allowedType in k for allowedType in allowedFileTypes) ]

    # A simple 1 liner to extract only the files out of the chosen directory
    def getFilesInDir(self, directory)-> list: 
        return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

    # Python is dumb and doesn't understand ../ so I made something more complicated to arbitrarily get the correct directory path
    def getImageFilesDir(self) -> str:
        # Declare constants
        rootDirName = 'DurOrNoDur' # Git root directory so it's fine
        currentFileDir = getcwd()

        # Format the directory structure to point to ImageFiles regardness of format/start location
        sliceIndex = currentFileDir.index(rootDirName)
        rootDir = currentFileDir[0 : sliceIndex + len(rootDirName) + 1]
        rootDir = rootDir[0:len(rootDir)-1] if rootDir[-1] == "\\" else  rootDir 
        return f"{rootDir}\ImageFiles"

    # Iterates over the image files to extract out the feature vectors
    def setAllFeatureVectors(self, imageFileNames, label) -> None:
        for imageName in imageFileNames:
            featureVectors = self.parseFeatureVectors(imageName)
            self.imageDict[imageName] = Image(imageName, label, featureVectors)

    # Extract out the feature vectors for a given image file
    def parseFeatureVectors(self, imageName) -> list:
        ## Put code here to store the feature vector as a list
        return [1, 2, 3]
        
