from imageParser import ImageParser 
from image import Image
from analysis import Analysis 
from classifier import Classifier 

### Import the files and extract out the feature vectors/labels from the data
myParser = ImageParser()
loadFromMemory = True
fileDict = myParser.loadImageDictFromFile() if loadFromMemory else myParser.constructDictionary()

### Use the image data dictionary to classify the data
myClassifier = Classifier(fileDict)

### Use the classifier to predict some data
#resultsKNN = myClassifier.KNNClassify()
resultsSVM = myClassifier.SVMClassify() 
### Put the output of our classifier for the test data in a nice output
myAnalysis = Analysis()
myAnalysis.graphResults(resultsSVM)