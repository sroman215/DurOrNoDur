from os import listdir, getcwd
from os.path import isfile, join
from image import Image
import glob
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image as img_format
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
        recursiveFilesAndFolders = glob.glob(directory + '/**/*', recursive=True)
        return [ list(filter(lambda f: "." in f, recursiveFilesAndFolders))[0] ] # Filters out the folders from the recursive lookup

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



    # Gaussian Kernel to be applied to the images for denoising
    def gaussian_kernel(self, size, sigma=1) -> list:
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    # Get Edges from image 
    # might need this line for image processing: from scipy import ndimage

    def sobel_filters(self,image) :
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = ndimage.filters.convolve(image, Kx)
        Iy = ndimage.filters.convolve(image, Ky)
        
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        
        return (G, theta)

    # Pull thge contents of the image based on the path
    def extractImage(self, imagePath):
        content = img_format.open(imagePath)
        img_resized = content.resize((200,200), img_format.ANTIALIAS)
        return img_resized

    # Extract out the feature vectors for a given image file
    def parseFeatureVectors(self, imagePath) -> list:
        ## Put code here to store the feature vector as a list
        content = self.extractImage(imagePath)
        content_array  = np.array(content)
        lowpass = ndimage.gaussian_filter(content_array, 10)
        gauss_highpass = content_array - lowpass
        plt.imshow(gauss_highpass[:,:,0])

        plt.show()    
        R = []
        B = []
        G = []
        for i in range(gauss_highpass.shape[0]):
            for j in range(gauss_highpass.shape[1]):
                R = np.append(R, gauss_highpass[i, j][2])
                G = np.append(G, gauss_highpass[i, j][1])
                B = np.append(B, gauss_highpass[i, j][0])
            
        
        #feature_vector = 
        return np.append(R,G,B) 
        
