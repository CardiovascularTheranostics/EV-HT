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
    return data, output, patients, output_0

def create_samples(data, output, patients, output_0):
    current_patient = -1
    samples = []
    output_labels = []
    patients_new = []
    for i in range(len(np.asarray(patients))):
        if current_patient != patients[i]:
            current_patient = patients[i]
        else:
            samples.append(np.concatenate((np.asarray(output[i-1]).reshape(1),(data[i]-data[i-1])/data[i-1]*100),axis=0))
            output_labels.append(output[i])
            patients_new.append(patients[i])
    return samples, output_labels, patients_new

def create_samples_R0(data, output, patients, output_0):
    for i in np.unique(patients):
        dati_paziente = data[patients==i]
        dati_paziente_R0_average = np.mean(data[np.logical_and(patients==i,output_0==0)], axis=0)
        dati_paziente = (dati_paziente-dati_paziente_R0_average)/dati_paziente_R0_average
        data[patients==i] = dati_paziente
    return data, output

def create_samples_temporal_order_R0(data, output, patients, output_0):
    current_patient = -1
    samples = []
    output_labels = []
    index_patient = 0
    patients_new = []
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
                samples.append((data[i]-dati_paziente_R0_average)/dati_paziente_R0_average*100)
                output_labels.append(output[i])
                patients_new.append(patients[i])
    return samples, output_labels, patients_new



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
    
def Patientfold_validation(X,Y, patients, df = None, n_tree = 30, max_leaf = 20, oversampling = 'SMOTE', model = 'LDA'):
    Accuracy = 0
    prediction = np.asarray([])
    y_prova = np.asarray([])
    np.random.seed(1)
    for i in np.unique(patients):
        X_train, X_test = X[patients!=i], X[patients==i]
        y_train, y_test = Y[patients!=i], Y[patients==i]
        model = RandomForestRegressor(n_estimators =n_tree,min_samples_leaf=10, max_leaf_nodes=max_leaf,random_state=1)

            
        # X_train, y_train = SMOTE(k_neighbors = 3).fit_resample(X_train, y_train)
        model0 = Imbalance_classes(X_train, y_train)
        X_train, y_train = model0.apply(oversampling)
        model.fit( X_train, y_train)
        # Acc = model.score(X_test,y_test)
        Acc = sum(((model.predict(X_test)>0.5).astype(int))==y_test)/len(y_test)
        prediction = np.concatenate((prediction,((model.predict(X_test))>0.5).astype(int)),axis=0)
        y_prova = np.concatenate((y_prova,y_test),axis=0)
        Accuracy = Accuracy + Acc
        #metrics to evaluate the algorithm. See the corresponding class.
    Accuracy = Accuracy/float(len(np.unique(patients)))
    index = y_prova==1
    sensitivity = np.mean(prediction[index]==1)*100
    spec = np.mean(prediction[~index]==0)*100
    print ('AccuracyPatientFOLD = %.2f \n' % (Accuracy*100))
    print(sk.metrics.confusion_matrix(np.asarray(y_prova).flatten(), np.asarray(prediction).flatten(), labels=None))
    df1 = pd.DataFrame({'Classification Algorithm':['RM'], 'Oversampling Method':[oversampling],  'n_tree':[n_tree], 'max_leaf':[max_leaf],  'Sensitvity':[sensitivity], 'Specificity': [spec], 'Accuracy':[Accuracy]})
    df = df.append(df1)
    return df

def launch_training(method,oversampling, samples,  output_labels, patients, df, n_tree = 50, max_leaf = 20):
    df = Patientfold_validation(samples, output_labels, patients, df, n_tree, max_leaf, oversampling)
    return df

def Kfold_validation(X,Y, model = 'LDA'):
    NFOLDS = X.shape[0]
    NFOLDS = 10
    np.random.seed(1)
    kf = KFold(n_splits=NFOLDS,shuffle=True, random_state=10)
    Accuracy = 0
    prediction = np.asarray([])
    y_prova = np.asarray([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model = RandomForestClassifier(n_estimators =30,min_samples_leaf=10, max_leaf_nodes=20,random_state=1)
        X_train, y_train = SMOTE(k_neighbors = 3).fit_resample(X_train, y_train)
        model.fit( X_train, y_train)
        Acc = model.score(X_test,y_test)
        prediction = np.concatenate((prediction,(model.predict(X_test))),axis=0)
        y_prova = np.concatenate((y_prova,y_test),axis=0)
        Accuracy = Accuracy + Acc
        #metrics to evaluate the algorithm. See the corresponding class.
    Accuracy = Accuracy/float(NFOLDS)
    print ('AccuracyKFOLD = %.2f \n' % (Accuracy*100))
    print(sk.metrics.confusion_matrix(np.asarray(y_prova).flatten(), np.asarray(prediction).flatten(), labels=None))
        
data, output, patients, output_0 = creation_dataset()
# samples, output_labels = create_samples_R0(data, output, patients, output_0)
# samples, output_labels, patients = create_samples(data, output, patients, output_0)
samples, output_labels, patients = create_samples_temporal_order_R0(data, output, patients, output_0)
samples = np.asarray(samples)
output_labels = np.asarray(output_labels)
np.random.seed(0)
# Kfold_validation(samples, output_labels, "RF")
# Patientfold_validation(samples, output_labels, patients, "RF")

methods = ['RF']
oversamplings = ['None','SMOTE', 'SMOTEENN', 'RandomOverSampling']
grid_parameters = {'n_tree': [10,50,100,200,400,800], 'max_leaf': [10,20,40,80,160,320]}
# grid_parameters = {'n_tree': [10,50,100], 'max_leaf': [10,20,40,80]}
# methods = ['RF']
# oversamplings = ['SMOTE']
# grid_parameters = {'n_tree': [30], 'max_leaf': [20]}
for method in methods:
    df = pd.DataFrame(columns=['Classification Algorithm', 'Oversampling Method', 'n_tree', 'max_leaf', 'Train/Validation [1/0]', 'Sensitvity', 'Specificity', 'Accuracy'])
    for oversampling in oversamplings:
        for tree in grid_parameters['n_tree']:
            for leaf in grid_parameters['max_leaf']:
                df = launch_training(method,oversampling, samples, output_labels, patients, df, n_tree = tree, max_leaf = leaf)
    df.to_excel("Grid_search.xlsx")