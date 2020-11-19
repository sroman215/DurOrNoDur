class Classifier:
    def __init__(self, data):
        self.data = data
        self.durData = self.splitDataByLabel(1)
        self.noDurData = self.splitDataByLabel(0)

    def splitDataByLabel(self, label): 
        newDict = dict()
        for (key, value) in self.data.items():
            if (value.label == label):
                newDict[key] = value
        return newDict

    def KNNClassify(self):
        return []

    def SVMClassify(self):
        return

    def CNNClassify(self):
        return

    def KNNPredict(self):
        return

    def SVMPredict(self):
        return

    def CNNPredict(self):
        return