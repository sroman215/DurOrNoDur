from imageParser import ImageParser 
from image import Image
from analysis import Analysis 
from classifier import Classifier 

### Import the files and extract out the feature vectors/labels from the data
myParser = ImageParser()
loadFromMemory = True
fileDict = myParser.loadImageDictToFile() if loadFromMemory else myParser.constructDictionary()

### Use the image data dictionary to classify the data
myClassifier = Classifier(myParser.imageDict)

### Use the classifier to predict some data
results = myClassifier.KNNClassify()

### Put the output of our classifier for the test data in a nice output
myAnalysis = Analysis()
myAnalysis.graphResults(results)