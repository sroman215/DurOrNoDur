from sklearn import neighbors

from sklearn import svm

import numpy as np

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
    
    def constructDataSet(self, start,stop):
        return [*list(self.durData.values())[start:stop], *list(self.noDurData.values())[start:stop]]

    def KNNClassify(self):
        maxNeighbors = 10
        Empircal_risk_array = [0 for x in range(maxNeighbors)]

        True_risk_array = [0 for x in range(maxNeighbors)] 

        
        for k in range(1, maxNeighbors):
            num_neighbors = k
            data_ratio_train = 0.65
            cutoff = int(np.floor(data_ratio_train*len(self.data)*data_ratio_train))
            test = len(self.data)
            trainData = self.constructDataSet(0,cutoff)
            lengthOfTrainData = len(trainData)
            fvsTrain = list(map(lambda x: x.featureVectors, trainData))
            labelsTrain = list(map(lambda x: x.label, trainData))
            lengthOfTrainData = len(labelsTrain)
            testData = self.constructDataSet(cutoff+1,int(np.floor(len(self.data))))
            fvsTest = list(map(lambda x: x.featureVectors, testData))
            labelsTest = list(map(lambda x: x.label, testData))
            lengthOfTrainData = len(labelsTest)

            print(k)

            KNNClassifier = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors,algorithm="brute")
            KNNClassifier.fit(fvsTrain,labelsTrain)
            predicted_labels = KNNClassifier.predict(fvsTest)
            predicted_empirical_values = (KNNClassifier.predict(fvsTrain))
            EMP_error=(predicted_empirical_values!=labelsTrain).sum()
            True_error = (predicted_labels!=labelsTest).sum()
            N = len(predicted_empirical_values)
            N_t = len(predicted_labels)
            Empircal_risk_array[k-1] = EMP_error/N
            True_risk_array[k-1] = True_error/N_t
            
        return np.array(Empircal_risk_array,True_risk_array)

    def SVMClassify(self):
        gamma_const=0.001
        C_const =10

        kernel_string = ['rbf','linear']
        Empircal_risk_array = [0 for x in range(len(kernel_string))]

        True_risk_array = [0 for x in range(len(kernel_string))] 

        data_ratio_train = 0.65
        cutoff = int(np.floor(data_ratio_train*len(self.data)*data_ratio_train))
        test = len(self.data)
        trainData = self.constructDataSet(0,cutoff)
        lengthOfTrainData = len(trainData)
        fvsTrain = list(map(lambda x: x.featureVectors, trainData))
        labelsTrain = list(map(lambda x: x.label, trainData))
        lengthOfTrainData = len(labelsTrain)
        testData = self.constructDataSet(cutoff+1,int(np.floor(len(self.data))))
        fvsTest = list(map(lambda x: x.featureVectors, testData))
        labelsTest = list(map(lambda x: x.label, testData))
        lengthOfTrainData = len(labelsTest)

        counter = 0

        for k in kernel_string:

            svmClass = svm.SVC(kernel=k, random_state=1, gamma=gamma_const, C=C_const,class_weight={1: 10})
            svmClass.fit(fvsTrain, labelsTrain)
            predicted_labels = svmClass.predict(fvsTest)
            predicted_empirical_values = (svmClass.predict(fvsTrain))
            EMP_error=(predicted_empirical_values!=labelsTrain).sum()
            True_error = (predicted_labels!=labelsTest).sum()
            N = len(predicted_empirical_values)
            N_t = len(predicted_labels)
            Empircal_risk_array[counter] = EMP_error/N
            True_risk_array[counter] = True_error/N_t
            counter = counter +1
        return np.array(Empircal_risk_array,True_risk_array)


    def CNNClassify(self):
        return

    def KNNPredict(self):
        return

    def SVMPredict(self):
        return

    def CNNPredict(self):
        return