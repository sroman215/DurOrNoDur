# An object to house and manipulate the data itself. May or may not end up being used
class Image:
    def __init__(self, fileName, label, featureVectors):
        self.fileName = fileName
        self.label = label
        self.featureVectors = featureVectors