class Classifier:
    def __init__(self, data):
        self.data = data
        return

    def KNNClassify(self,num_neighbors,image_train,image_test):
        from sklearn import neighbors
        KNNClassifier = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors,algorithm="brute")
        KNNClassifier.fit(image_train.featureVectors,image_train.label)
        KNNClassifier.fit(image_train.featureVectors,image_train.label)
        predicted_labels = KNNClassifier.predict(image_test.featureVectors)
        predicted_empirical_values = (KNNClassifier.predict(image_train.label))
        predicted_test_values =  (KNNClassifier.predict(test_set_data))
        EMP_error=(predicted_empirical_values!=train_set_label).sum()
        True_error = (predicted_test_values!=test_set_label).sum()
        N = len(predicted_empirical_values)
        N_t = len(predicted_test_values)
        Empircal_risk_array = EMP_error/N
        True_risk_array = True_error/N_t
        
        return np.array(Empircal_risk_array,True_risk_array)

    def SVMClassify(self,image_train,image_test,kernel_string,gamma_const,C_const):
        from sklearn.svm import SVC
        svm = SVC(kernel=kernel_string, random_state=1, gamma=gamma_const, C=C_const)
        svm.fit(image_train.featureVectors, image_train.label)
        predicted_labels = svm.predict(image_test.featureVectors)
        predicted_empirical_values = (svm.predict(image_train.label))
        predicted_test_values =  (svm.predict(test_set_data))
        EMP_error=(predicted_empirical_values!=train_set_label).sum()
        True_error = (predicted_test_values!=test_set_label).sum()
        N = len(predicted_empirical_values)
        N_t = len(predicted_test_values)
        Empircal_risk_array = EMP_error/N
        True_risk_array = True_error/N_t
        return np.array(Empircal_risk_array,True_risk_array)


    def CNNClassify(self):
        return

    def KNNPredict(self):
        return

    def SVMPredict(self):
        return

    def CNNPredict(self):
        return