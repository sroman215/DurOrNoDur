from imageParser import ImageParser 
from image import Image
from analysis import Analysis 
from classifier import Classifier 

### Import the files and extract out the feature vectors from the data
myParser = ImageParser()
print(myParser.featureVectorDict)

### Parse the feature vector dictionary into feature vectors, then run the classifier 
featureVectors = myParser.featureVectorDict.values()
myClassifier = Classifier(featureVectors)

### Use the classifier to predict some data
results = myClassifier.predict()

### Put the output of our classifier for the test data in a nice output
myAnalysis = Analysis()
myAnalysis.graphResults(results)