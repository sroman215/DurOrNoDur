# An object to house file name, label, and feature vectors. May be used for some feature vector calculation/manipulation
class Image:
    def __init__(self, fileName, label, featureVectors):
        self.fileName = fileName
        self.label = label
        self.featureVectors = featureVectors

    def printValues(self):
        print(f"File Name: {self.fileName}, Label: {self.label}, Feature Vectors: {self.featureVectors}")