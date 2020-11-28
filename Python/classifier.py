from sklearn import neighbors

from sklearn import svm

import numpy as np
from os import listdir, getcwd
from os.path import isfile, join

class Classifier:
    KNNEmpRisk = []
    KNNTrueRisk= []
    KNNPredictedLabels =[]
    KNNEmpPredictedLabels = []
    KNNMissedDeer = []
    KNNMissedDeerTrue = []
    SVCEmpRisk = []
    SVCTrueRisk= []
    SVCPredictedLabels =[]
    SVCEmpPredictedLabels = []
    SVCMissedDeer = []
    SVCMissedDeerTrue =  []
    SVCLinearRisk = []
    SVCLinearTrueRisk = []
    SVCLinearMissedDeer = []
    SVCLinearMissedDeerTrue =[]
    SVCRBFRisk = []
    SVCRBFTrueRisk = []
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
        
    def loadImageEmpRiskFromFileKNN(self):
        return np.load(join(f"{self.getRootDir()}\\Python", self.KNNEmpRisk), allow_pickle='TRUE').item()

    def loadImageEmpLabelsKNNFromFile(self) :
        return np.load(join(f"{self.getRootDir()}\\Python", self.KNNEmpPredictedLabels), allow_pickle='TRUE').item()

    def loadImageTrueRiskFromFileKNN(self):
        return np.load(join(f"{self.getRootDir()}\\Python", self.KNNTrueRisk), allow_pickle='TRUE').item()

    def loadImagePredictedLabelsKNNFromFile(self) :
        return np.load(join(f"{self.getRootDir()}\\Python", self.KNNPredictedLabels), allow_pickle='TRUE').item()
     
    # Load SVC results 

    def loadImageEmpRiskFromFileSVC(self):
        return np.load(join(f"{self.getRootDir()}\\Python", self.SVCEmpRisk), allow_pickle='TRUE').item()

    def loadImageEmpLabelsFromFileSVC(self) :
        return np.load(join(f"{self.getRootDir()}\\Python", self.SVCEmpPredictedLabels), allow_pickle='TRUE').item()
     
    def loadImageTrueRiskFromFileSVC(self) : 
        return np.load(join(f"{self.getRootDir()}\\Python", self.SVCTrueRisk), allow_pickle='TRUE').item()

    def loadImagePredictedLabelsromFileSVC(self) :
        return np.load(join(f"{self.getRootDir()}\\Python", self.SVCPredictedLabels), allow_pickle='TRUE').item()
     


    def constructDataSet(self, start,stop):
        return [*list(self.durData.values())[start:stop], *list(self.noDurData.values())[start:stop]]

    def saveStatsToFile(self,resultsFileName, obj) -> None:
        np.save(join(f"{self.getRootDir()}\\Python", resultsFileName), obj)
        # Python is dumb and doesn't understand ../ so I made something more complicated to arbitrarily get the correct directory path
    
    def getRootDir(self) -> str:
        # Declare constants
        rootDirName = 'DurOrNoDur' # Git root directory so it's fine
        currentFileDir = getcwd()

        # Format the directory structure to point to ImageFiles regardness of format/start location
        sliceIndex = currentFileDir.index(rootDirName)
        rootDir = currentFileDir[0 : sliceIndex + len(rootDirName) + 1]
        return rootDir[0:len(rootDir)-1] if rootDir[-1] == "\\" else  rootDir 

    def KNNClassify(self):
        maxNeighbors = 20
        predicted_empirical_values =[]
        predicted_labels =[]
        MissedDeerVector =[0 for x in range((maxNeighbors-1))]
        MissedDeerVectorTrue = [0 for x in range((maxNeighbors-1))]
        num_deer = 0
        Empircal_risk_array = [0 for x in range(maxNeighbors-1)]
        numMissedDeer = 0
        True_risk_array = [0 for x in range(maxNeighbors-1)] 
        numDeerTrue = 0
        numMissedDeerTrue = 0
        num_deer_True = 0
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
            tempPredEmp = KNNClassifier.predict(fvsTrain)
            tempPredTrue = KNNClassifier.predict(fvsTest)
            predicted_labels = np.append(predicted_labels,KNNClassifier.predict(fvsTest))
            predicted_empirical_values = np.append(predicted_empirical_values,(KNNClassifier.predict(fvsTrain)))
            EMP_error=(tempPredEmp!=labelsTrain).sum()
            True_error = (tempPredTrue!=labelsTest).sum()
            N = len(tempPredEmp)
            N_t = len(tempPredTrue)
            for i in range(len(labelsTrain)):
                if labelsTrain[i] ==1:
                    num_deer = num_deer+1
                    if labelsTrain[i]!=predicted_empirical_values[i]:
                        numDeerTrue = numDeerTrue +1

            for i in range(len(labelsTest)):
                if labelsTest[i] ==1:

                    num_deer_True =  num_deer_True+1
                    if labelsTest[i] !=predicted_labels[i]:
                        numMissedDeerTrue =numMissedDeerTrue+1 
                
            
            MissedDeerVector[k-1] = numMissedDeer/num_deer
            MissedDeerVectorTrue[k-1] = numMissedDeerTrue/num_deer_True
            Empircal_risk_array[k-1] = EMP_error/N
            True_risk_array[k-1] = True_error/N_t
            
        self.saveStatsToFile(self.KNNEmpRisk,Empircal_risk_array)
        self.saveStatsToFile(self.KNNTrueRisk,True_risk_array)
        self.saveStatsToFile(self.KNNPredictedLabels,predicted_labels)
        self.saveStatsToFile(self.KNNEmpPredictedLabels,predicted_empirical_values)
        self.saveStatsToFile(self.KNNMissedDeer,MissedDeerVector)        
        self.saveStatsToFile(self.KNNMissedDeerTrue, MissedDeerVectorTrue)
        return []

    def SVMClassify(self):
        gamma_const=0.001
        C_const =10
        predicted_empirical_values =[]
        predicted_labels =[]

        kernel_string = ['rbf','linear']

        Empircal_risk_array = [0 for x in range(2)]

        True_risk_array = [0 for x in range(2)] 

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
            tempPredEmp = svmClass.predict(fvsTrain)
            tempPredTrue = svmClass.predict(fvsTest)
            predicted_labels = np.append(predicted_labels,svmClass.predict(fvsTest))
            predicted_empirical_values = np.append(predicted_empirical_values,(svmClass.predict(fvsTrain)))
            EMP_error=(tempPredEmp!=labelsTrain).sum()
            True_error = (tempPredTrue!=labelsTest).sum()
           
            N = len(tempPredEmp)
            N_t = len(tempPredTrue)
            Empircal_risk_array[counter] = EMP_error/N
            True_risk_array[counter] = True_error/N_t
            counter = counter+1
        self.saveStatsToFile(self.SVCEmpRisk,Empircal_risk_array)
        self.saveStatsToFile(self.SVCTrueRisk,True_risk_array)
        self.saveStatsToFile(self.SVCPredictedLabels,predicted_labels)
        self.saveStatsToFile(self.SVCEmpPredictedLabels,predicted_empirical_values)
        
        return[]
    def SVMClassifyRBF_SweepC(self):
            #gamma_const=0.001
            C_array =[10,100,1000,10000]
            predicted_empirical_values =[]
            predicted_labels =[]

            kernel_string = ['rbf']
            numMissedDeer = 0
            Empircal_risk_array = [0 for x in range(len(C_array))]
            MissedDeerVector =[0 for x in range(len(C_array))]
            True_risk_array = [0 for x in range(len(C_array))] 
            MissedDeerVectorTrue =[0 for x in range(len(C_array))]

            num_deer = 0
            numDeerTrue = 0
            num_deer_True = 0
            numMissedDeerTrue = 0
            
            
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
            print('RBF')
            for k in C_array:
                print(k)

                svmClass = svm.SVC(kernel='rbf', random_state=1, C=k )
                svmClass.fit(fvsTrain, labelsTrain)
                tempPredEmp = svmClass.predict(fvsTrain)
                tempPredTrue = svmClass.predict(fvsTest)
                predicted_labels = np.append(predicted_labels,svmClass.predict(fvsTest))
                predicted_empirical_values = np.append(predicted_empirical_values,(svmClass.predict(fvsTrain)))
                EMP_error=(tempPredEmp!=labelsTrain).sum()
                True_error = (tempPredTrue!=labelsTest).sum()
                
                for i in range(len(labelsTrain)):
                    if labelsTrain[i] ==1:
                        num_deer = num_deer+1
                        if labelsTrain[i]!=predicted_empirical_values[i]:
                            numMissedDeer = numMissedDeer +1
                        
                for i in range(len(labelsTest)):
                    if labelsTest[i] ==1:
                        num_deer_True =num_deer_True+1 
                        if labelsTest[i]!=predicted_labels[i]:
                            numMissedDeerTrue = numMissedDeerTrue+1
                
                MissedDeerVector[counter] = numMissedDeer/num_deer
                MissedDeerVectorTrue[counter] = numMissedDeerTrue/num_deer_True
                N = len(tempPredEmp)
                N_t = len(tempPredTrue)
                Empircal_risk_array[counter] = EMP_error/N
                True_risk_array[counter] = True_error/N_t
                
                
                counter = counter+1


            self.saveStatsToFile(self.SVCRBFRisk,Empircal_risk_array)

            self.saveStatsToFile(self.SVCRBFTrueRisk,True_risk_array)
            self.saveStatsToFile(self.SVCMissedDeerTrue,MissedDeerVector)
            self.saveStatsToFile(self.SVCMissedDeerTrue,MissedDeerVectorTrue)
            return[]
    def SVMClassifyLinear_SweepC(self):
            #gamma_const=0.001
            C_array =[10,100,1000,10000]
            predicted_empirical_values =[]
            predicted_labels =[]

            kernel_string = ['linear']
            numMissedDeer = 0
            Empircal_risk_array = [0 for x in range(len(C_array))]
            MissedDeerVector =[0 for x in range(len(C_array))]
            True_risk_array = [0 for x in range(len(C_array))] 
            MissedDeerVectorTrue =[0 for x in range(len(C_array))]

            num_deer = 0
            numDeerTrue = 0
            num_deer_True = 0
            numMissedDeerTrue = 0
            
            
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
            print('Linear')
            for k in C_array:
                print(k)
                svmClass = svm.SVC(kernel='linear', random_state=1, C=k )
                svmClass.fit(fvsTrain, labelsTrain)
                tempPredEmp = svmClass.predict(fvsTrain)
                tempPredTrue = svmClass.predict(fvsTest)
                predicted_labels = np.append(predicted_labels,svmClass.predict(fvsTest))
                predicted_empirical_values = np.append(predicted_empirical_values,(svmClass.predict(fvsTrain)))
                EMP_error=(tempPredEmp!=labelsTrain).sum()
                True_error = (tempPredTrue!=labelsTest).sum()
                
                for i in range(len(labelsTrain)):
                    if labelsTrain[i] ==1:
                        num_deer = num_deer+1
                        if labelsTrain[i]!=predicted_empirical_values[i]:
                            numMissedDeer = numMissedDeer +1
                        
                for i in range(len(labelsTest)):
                    if labelsTest[i] ==1:
                        num_deer_True =num_deer_True+1 
                        if labelsTest[i]!=predicted_labels[i]:
                            numMissedDeerTrue = numMissedDeerTrue+1
                
                MissedDeerVector[counter] = numMissedDeer/num_deer
                MissedDeerVectorTrue[counter] = numMissedDeerTrue/num_deer_True
                N = len(tempPredEmp)
                N_t = len(tempPredTrue)
                Empircal_risk_array[counter] = EMP_error/N
                True_risk_array[counter] = True_error/N_t
                
                
                counter = counter+1


            self.saveStatsToFile(self.SVCLinearRisk,Empircal_risk_array)

            self.saveStatsToFile(self.SVCLinearTrueRisk,True_risk_array)
            self.saveStatsToFile(self.SVCLinearMissedDeerTrue,MissedDeerVector)
            self.saveStatsToFile(self.SVCLinearMissedDeerTrue,MissedDeerVectorTrue)
            return[]

def CNNClassify(self):
    return
def KNNPredict(self):
    return

def SVMPredict(self):
    return

def CNNPredict(self):
     return