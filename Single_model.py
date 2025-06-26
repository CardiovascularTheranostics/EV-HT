from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn import svm
import pdb
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

from matplotlib import rc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn as sk
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN
import copy 

# activate latex text rendering
# rc('text', usetex=True)

def creation_dataset():
    sheet = 'FC peptide EV surface Profiling'
    df = pd.read_excel('EM-HT Longitudinal - Db 2023.07.25_JB.xlsx', sheet_name = sheet)
    data = df.values
    output_0 = copy.deepcopy(df["Reject ANALYSIS (0=n0; 1=grade1; 2=grade2; 3=grade3)"].values)
    output = df["Reject ANALYSIS (0=n0; 1=grade1; 2=grade2; 3=grade3)"]
    output = output.values
    output[output==1]=0
    output[output==2]=1
    output[output==3]=1
    data = data[:,81:96]
    patients = df["Patient"]
    patients_names = patients.unique()
    patients = patients.values
    for i, name in enumerate(patients_names):
        patients[patients==name] = i
    return data, output, patients, output_0, df.keys()[81:96], df["Study ID"]

def creation_dataset_validation():
    sheet = 'MACSPlex ML Validation'
    df = pd.read_excel('EM-HT Longitudinal - Db 2023.07.25_JB.xlsx', sheet_name = sheet)
    data = df.values
    output_0 = copy.deepcopy(df["Reject ANALYSIS (0=n0; 1=grade1; 2=grade2; 3=grade3)"].values)
    output = df["Reject ANALYSIS (0=n0; 1=grade1; 2=grade2; 3=grade3)"]
    output = output.values
    output[output==1]=0
    output[output==2]=1
    output[output==3]=1
    data = data[:,81:96]
    patients = df["Patient"]
    patients_names = patients.unique()
    patients = patients.values
    for i, name in enumerate(patients_names):
        patients[patients==name] = i
    return data, output, patients, output_0, df.keys()[81:96], df["Study ID"]



def create_samples_temporal_order_R0(data, output, patients, output_0, studyID):
    current_patient = -1
    samples = []
    output_labels = []
    index_patient = 0
    patients_new = []
    studyID_new = []
    for i in range(len(np.asarray(patients))):
        if current_patient != patients[i]:
            current_patient = patients[i]
            index_patient = 0
        else:
            index_patient +=1
            dati_paziente = data[patients==patients[i]]
            output_paziente = output_0[patients==patients[i]]
            dati_paziente = dati_paziente[:index_patient]
            output_paziente = output_paziente[:index_patient]
            if len(dati_paziente[output_paziente==0]) > 0:
                dati_paziente_R0_average = np.mean(dati_paziente[output_paziente==0], axis=0)
                try:
                    samples.append((data[i]-dati_paziente_R0_average)/dati_paziente_R0_average*100)
                    output_labels.append(output[i])
                    patients_new.append(patients[i])
                    studyID_new.append(studyID[i])
                except:
                    continue
    return samples, output_labels, patients_new, studyID_new



class Imbalance_classes:
    def __init__(self, xtrain,ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain

    def apply(self,string):
        #Description: the method resample the dataset with the selected method, giving as output the resampled dataset.
        #INPUT:  - string: used to choose the selected thecnique
        #OUTPUT: - X_train: new dataset resampled with the choosen algorithm.
        #        - y_train: labels corresponding to the new dataset.
        #print ('Dimensional Reduction')
        #print ('---------------------')
        if string == 'SMOTE':
            tl = SMOTE(k_neighbors = 3,random_state=1)
        elif string == 'SMOTEENN':
            #3 neighbors for Edited nearest neighbors: Normal SMOTE: distance of 2 points of the class and new point in between
            tl = SMOTEENN(random_state=42)
        elif string == 'RandomOverSampling':
            tl = RandomOverSampler(random_state=1)
        elif string == 'None':
            return self.xtrain, self.ytrain
        X_train, y_train = tl.fit_resample(self.xtrain, self.ytrain)
        return X_train, y_train
    
def Patientfold_validation(X,Y, patients, labels, studyID, oversampling = 'RandomOverSampling'):
    Accuracy = 0
    prediction = np.asarray([])
    y_prova = np.asarray([])
    study_ID = np.asarray([])
    studyID = np.asarray(studyID)
    prediction_probabilities = np.asarray([])
    np.random.seed(1)
    for i in np.unique(patients):
        X_train, X_test = X[patients!=i], X[patients==i]
        y_train, y_test = Y[patients!=i], Y[patients==i]
        studyID_test = studyID[patients==i]
        model = RandomForestRegressor(n_estimators =10,min_samples_leaf=10, max_leaf_nodes=30,random_state=1)
        model0 = Imbalance_classes(X_train, y_train)
        X_train, y_train = model0.apply(oversampling)
        model.fit( X_train, y_train)
        if i == 0 and False:
            for alb in np.arange(10):
                estimator = model.estimators_[alb]
                from sklearn.tree import export_graphviz

                export_graphviz(estimator, out_file='tree37.dot',
                                feature_names = labels,
                                rounded = True, proportion = False,
                                precision = 2, filled = True)
                # Convert to png using system command (requires Graphviz)
                import os
                os.system('dot -Tpng tree37.dot -o ./Alberi/tree_' + str(alb)+ '.png')
        Acc = sum(((model.predict(X_test)>0.5).astype(int))==y_test)/len(y_test)
        prediction = np.concatenate((prediction,((model.predict(X_test))>0.5).astype(int)),axis=0)
        y_prova = np.concatenate((y_prova,y_test),axis=0)
        prediction_probabilities = np.concatenate((prediction_probabilities,model.predict(X_test)),axis=0)
        study_ID = np.concatenate((study_ID,studyID_test),axis=0)
        Accuracy = Accuracy + Acc
        #metrics to evaluate the algorithm. See the corresponding class.
    Accuracy = Accuracy/float(len(np.unique(patients)))
    index = y_prova==1
    sensitivity = np.mean(prediction[index]==1)*100
    spec = np.mean(prediction[~index]==0)*100
    print ('AccuracyPatientFOLD = %.2f \n' % (Accuracy*100))
    print ('SensitivityPatientFOLD = %.2f \n' % (sensitivity))
    print ('SpecificityPatientFOLD = %.2f \n' % (spec))
    print(sk.metrics.confusion_matrix(np.asarray(y_prova).flatten(), np.asarray(prediction).flatten(), labels=None))
    prova =pd.DataFrame([prediction_probabilities, study_ID])
    prova.to_excel("probabilities.xlsx")
    
    
def Patientfold_validation_MACsplex(X,Y, patients, X_validation, Y_validation, studyID, oversampling = 'RandomOverSampling'):
    Accuracy = 0
    prediction = np.asarray([])
    y_prova = np.asarray([])
    study_ID = np.asarray([])
    studyID = np.asarray(studyID)
    prediction_probabilities = np.asarray([])
    np.random.seed(1)
    NFOLDS = X.shape[0]
    NFOLDS = 10
    np.random.seed(1)
    kf = KFold(n_splits=NFOLDS,shuffle=True, random_state=10)
    for train_index, test_index in kf.split(X):
    # for i in np.unique(patients):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # X_train, X_test = X[patients!=i], X_validation[patients==i]
        # y_train, y_test = Y[patients!=i], Y_validation[patients==i]
        studyID_test = studyID[test_index]
        # studyID_test = studyID[patients==i]
        ########## DIRE CHE NE ABBIAMO FATTI 10
        model = RandomForestRegressor(n_estimators =100,min_samples_leaf=10, max_leaf_nodes=20,random_state=1)
        model0 = Imbalance_classes(X_train, y_train)
        X_train, y_train = model0.apply(oversampling)
        model.fit( X_train, y_train)
        Acc = sum(((model.predict(X_test)>0.5).astype(int))==y_test)/len(y_test)
        prediction = np.concatenate((prediction,((model.predict(X_test))>0.5).astype(int)),axis=0)
        y_prova = np.concatenate((y_prova,y_test),axis=0)
        prediction_probabilities = np.concatenate((prediction_probabilities,model.predict(X_test)),axis=0)
        study_ID = np.concatenate((study_ID,studyID_test),axis=0)
        Accuracy = Accuracy + Acc
        #metrics to evaluate the algorithm. See the corresponding class.
    # Accuracy = Accuracy/float(len(np.unique(patients)))
    Accuracy = Accuracy/float(NFOLDS)
    index = y_prova==1
    sensitivity = np.mean(prediction[index]==1)*100
    spec = np.mean(prediction[~index]==0)*100
    print ('AccuracyPatientFOLD = %.2f \n' % (Accuracy*100))
    print ('SensitivityPatientFOLD = %.2f \n' % (sensitivity))
    print ('SpecificityPatientFOLD = %.2f \n' % (spec))
    print(sk.metrics.confusion_matrix(np.asarray(y_prova).flatten(), np.asarray(prediction).flatten(), labels=None))
    prova =pd.DataFrame([prediction_probabilities, study_ID])
    prova.to_excel("probabilities_validation.xlsx") 

        
data, output, patients, output_0, labels, studyID = creation_dataset()
data_validation, output_validation, patients_validation, output_0_validation, labels_validation, studyID_validation = creation_dataset_validation()
samples, output_labels, patients, studyID = create_samples_temporal_order_R0(data, output, patients, output_0, studyID)
samples_validation, output_labels_validation, patients_validation, studyID_validation = create_samples_temporal_order_R0(data_validation, output_validation, patients_validation, output_0_validation, studyID_validation)
samples = np.asarray(samples)
output_labels = np.asarray(output_labels)
samples_validation = np.asarray(samples_validation)
output_labels_validation = np.asarray(output_labels_validation)
np.random.seed(0)
# Kfold_validation(samples, output_labels, "RF")
Patientfold_validation(samples, output_labels, patients, labels, studyID)
Patientfold_validation_MACsplex(samples_validation, output_labels_validation, patients_validation, samples_validation, output_labels_validation, studyID_validation)
Patientfold_validation_MACsplex(samples, output_labels, patients, samples, output_labels, studyID)
