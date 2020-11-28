from imageParser import ImageParser 
from image import Image
from analysis import Analysis 
from classifier import Classifier 
import numpy as np
import matplotlib.pyplot as plt 
import math

### Import the files and extract out the feature vectors/labels from the data
myParser = ImageParser()
myParser.dictFileName = 'imageDictOutputHPNew.npy'
myParser.flagHP = True
myParserNoHP = ImageParser()
myParserNoHP.dictFileName = 'imageDictOutputNoHP.npy'
myParserNoHP.flagHP = False
loadFromMemory = True
loadFromMemoryNoHP = True
fileDict = myParser.loadImageDictFromFile() if loadFromMemory else myParser.constructDictionary()
fileDictNoHP = myParserNoHP.loadImageDictFromFile() if loadFromMemoryNoHP else myParserNoHP.constructDictionary()

### Use the image data dictionary to classify the data

# root = 'D:\\ML\\DurOrNoDur\\Python\\
# set files for results
myClassifier = Classifier(fileDict)
myClassifierNoHP = Classifier(fileDictNoHP)
myClassifier.KNNEmpRisk = 'KNNEmpRiskHP.npy'
myClassifier.KNNTrueRisk = 'KNNTrueRiskHP.npy'
myClassifier.KNNPredictedLabels = 'KNNPredLabelsHP.npy'
myClassifier.KNNEmpPredictedLabels = 'KNNEmpLabelsHP.npy'
myClassifier.KNNMissedDeer = 'KNNMissedDeerEmpHP.npy'
myClassifier.KNNMissedDeerTrue = 'KNNMissedDeerTrueHP.npy'


myClassifier.SVCEmpRisk = 'SVCEmpRiskHP.npy'
myClassifier.SVCTrueRisk = 'SVCTrueRiskHP.npy'
myClassifier.SVCPredictedLabels = 'SVCPredLabelsHP.npy'
myClassifier.SVCEmpPredictedLabels = 'SVCEmpLabelsHP.npy'
myClassifier.SVCMissedDeer = 'SVCMissedDeer.npy'
myClassifier.SVCMissedDeerTrue = 'SVCMissedDeerTrue.npy'



myClassifierNoHP.KNNEmpRisk = 'KNNEmpRiskNoHP.npy'
myClassifierNoHP.KNNTrueRisk = 'KNNTrueRiskNoHP.npy'
myClassifierNoHP.KNNPredictedLabels = 'KNNPredLabelsNoHP.npy'
myClassifierNoHP.KNNEmpPredictedLabels = 'KNNEmpLabelsNoHP.npy'
myClassifierNoHP.KNNMissedDeer = 'KNNMissedDeerEmpNoHP.npy'
myClassifierNoHP.KNNMissedDeerTrue = 'KNNMissedDeerTrueNoHP.npy'

myClassifierNoHP.SVCEmpRisk = 'SVCEmpRiskNoHP.npy'
myClassifierNoHP.SVCTrueRisk = 'SVCTrueRiskNoHP.npy'
myClassifierNoHP.SVCPredictedLabels = 'SVCPredLabelsNoHP.npy'
myClassifierNoHP.SVCEmpPredictedLabels = 'SVCEmpLabelsNoHP.npy'


myClassifier.SVCRBFRisk = 'SVCRBFRisk.npy'
myClassifier.SVCRBFRiskNoHP = 'SVCRBFRiskNoHP.npy'
myClassifier.SVCMissedDeer = 'SVCMissedDeer.npy'
myClassifier.SVCMissedDeerTrue = 'SVCMissedDeerTrue.npy'


myClassifier.SVCLinearRisk = 'SVCLinearRisk.npy'
myClassifier.SVCLinearRiskNoHP = 'SVCLinearRiskNoHP.npy'
myClassifier.SVCLinearMissedDeer = 'SVCLinearMissedDeer.npy'
myClassifier.SVCLinearMissedDeerTrue = 'SVCLinearMissedDeerTrue.npy'


### Use the classifier to predict some data
#resultsKNN = myClassifier.KNNClassify()
#resultsKNNNoHP = myClassifierNoHP.KNNClassify()

#resultsSVM = myClassifier.SVMClassify() 
#resultsSVMNoHP = myClassifierNoHP.SVMClassify() 
resultsSVM = myClassifier.SVMClassifyRBF_SweepC() 
resultsSVMLinear = myClassifier.SVMClassifyLinear_SweepC() 

# Load Results from KNN 
#root = 'D:\\ML\\DurOrNoDur\\Python\\'
EMPRiskKNNHP = np.load(myClassifier.KNNEmpRisk)
TrueRiskKNNHP= np.load(myClassifier.KNNTrueRisk)


#EMPLabelsKNNHP = np.load(myClassifier.KNNEmpPredictedLabels)
#TrueLabelsKNNHP = np.load(myClassifier.KNNPredictedLabels)

EMPRiskKNNNoHP= np.load(myClassifierNoHP.KNNEmpRisk)
TrueRiskKNNNoHP = np.load(myClassifierNoHP.KNNTrueRisk)

EMPLabelsKNNNoHP = np.load(myClassifierNoHP.KNNEmpPredictedLabels)
TrueLabelsKNNNoHP = np.load(myClassifierNoHP.KNNPredictedLabels)

# Load Results from SVC
EMPRiskSVCHP = np.load(myClassifier.SVCEmpRisk)
TrueRiskSVCHP= np.load(myClassifier.SVCTrueRisk)

#EMPLabelsKNNHP = np.load(myClassifier.KNNEmpPredictedLabels)
#TrueLabelsKNNHP = np.load(myClassifier.KNNPredictedLabels)

EMPRiskSVCNoHP= np.load(myClassifierNoHP.SVCEmpRisk)
TrueRiskSVCNoHP = np.load(myClassifierNoHP.SVCTrueRisk)

#EMPLabelsKNNNoHP = np.load(myClassifierNoHP.SVCEmpPredictedLabels)
#TrueLabelsKNNNoHP = np.load(myClassifierNoHP.SVCPredictedLabels)

knn_array = [x+1 for x in range(20-1)]
## plot data 
plt.plot(knn_array, EMPRiskKNNHP, 'r-',label = 'Empirical Risk w/HP')
plt.plot(knn_array, TrueRiskKNNHP, 'b--',label = 'True Risk w/HP')
plt.plot(knn_array, EMPRiskKNNNoHP, 'g-',label = 'Empirical Risk w/o HP')
plt.plot(knn_array,TrueRiskKNNNoHP,'y--',label = 'True Risk w/o HP')
plt.xlabel('K-Value')
plt.ylabel('Calculated Risk')
plt.legend(loc="lower right")
plt.title('KNN Risk Results')
xint = range(min(knn_array), math.ceil(max(knn_array))+1)

plt.xticks(xint)
plt.show()


### Put the output of our classifier for the test data in a nice output
myAnalysis = Analysis()
#myAnalysis.graphResults(resultsSVM)