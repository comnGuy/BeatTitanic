from classes.FeatureEngineering import FeatureEngineering
from classes.FeatureSelection import FeatureSelection
from classes.LogisticRegression import Logistic_Regression
from classes.RandomForest import RandomForest
from classes.SupportVectorMachines import SupportVectorMachines
from classes.VotingModels import VotingModels
from classes.NeuronalNet import NeuronalNet
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier

start_time = time.time()



# Read CSV
originTrainData = pd.read_csv('data/origin/train.csv')
originTestData = pd.read_csv('data/origin/test.csv')
originGenderSubmission = pd.read_csv('data/origin/gender_submission.csv')


# At the beginning we try to extract new features out of the data set
# for the training and the test set.

########### Feature Engineering ###########
# Initialise the feature enginnering
# @param trainings data
# @param tests data
# @param features for the linear model. Here you can add or remove features to try to get better results.
featureEngineering = FeatureEngineering({'originTrainData': originTrainData,
                                          'originTestData': originTestData,
                                          'originGenderSubmission': originGenderSubmission,
                                          'featuresLinearModel': ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']})

# First we try to predict the empty cells with our given featuresLinearModel
featureEngineering.linearPredictEmptyCells()
# Combine the linear information with the original data (Age, Fare)
featureEngineering.combineNaNCellWithLinearPrediction()
featureEngineering.normalize()
# Extract information out of the columns to generate new columns
featureEngineering.extractInformations()
# Create bins with various features
featureEngineering.createBins()
# Calcute family
featureEngineering.calculateFamily()
featureEngineering.SexPclass()
featureEngineering.SexAgeBin()

# Extract the title of the name
featureEngineering.extractPersonTitle()
# Remove Columns
featureEngineering.removeUnnecessaryColumns()
#Show sum of entries with NaNs
#featureEngineering.printNaNs()



#print(featureEngineering.trainData)


"""
argsFeatureSelection = {'trainData' : featureEngineering.trainData,
                        'testData': featureEngineering.testData}

fs = FeatureSelection(argsFeatureSelection)
fitScore = fs.univerateSelection()
fs.barPlot(fitScore)
ranking = fs.recursiveFeatureElimination()
fs.barPlot(ranking, ascending = False, columns='Ranking')
pcaComponents = fs.PCA()
fs.barPlot(pcaComponents[0], sort = False)
importance = fs.importanceExtraTrees()
fs.barPlot(importance)
fs.showPlots()



import sys
sys.exit()
"""

#print(featureEngineering.trainData)
#print(list(featureEngineering.trainData.index)[:-6])
#dtc = DecisionTreeClassifier(criterion='gini', random_state=7)
#fs.ablative_analysis(dtc, featureEngineering.trainData, featureEngineering.trainData.columns.values)





featureEngineering.removeVotedFeatures()
featureEngineering.printNaNs()


########### Random Forest ###########
"""argsRandomForest = {'trainData' : featureEngineering.trainData,
                    'testData': featureEngineering.testData,
                    'accuracyPathRandomForest': 'data/Accuracy/RandomForest/',
                    'numberTrees' : [5, 10, 50, 100],
                    'subsetSelection': {
                        1: {'features': ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'PersonTitle',
                                         'Bin_Age', 'Bin_Fare', 'Linear_Age', 'Linear_Fare', 'AgeAndLinear',
                                         'FareAndLinear', 'Family', '#Family', 'Male', 'Female'],
                            'filename': 'model_1.csv',
                            'loadFile': False}
                    #    2: {'features': ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'PersonTitle', 'Linear_Age', 'Linear_Fare'],
                    #        'filename': 'model_2.csv',
                    #        'loadFile': False},
                    #    3: {'features': ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'PersonTitle', 'AgeAndLinear', 'FareAndLinear'],
                    #        'filename': 'model_3.csv',
                    #        'loadFile': False},
                    #    4: {'features': ['Pclass', 'Sex', 'Family', '#Family', 'Embarked', 'PersonTitle', 'AgeAndLinear', 'FareAndLinear'],
                    #        'filename': 'model_4.csv',
                    #        'loadFile': False}
                        }
                    }
"""



# long
argsRandomForest = {'trainData' : featureEngineering.trainData,
                    'testData': featureEngineering.testData,
                    'crossValidation': 10,
                    'parameters': { 'n_estimators': [5, 15, 50, 100, 300, 500],
                                    'max_features': [3, 4, 5, 'auto'],
                                    'min_samples_leaf': [9, 10, 12],
                                    'random_state': [12345] }
                    }

"""
# short
argsRandomForest = {'trainData' : featureEngineering.trainData,
                    'testData': featureEngineering.testData,
                    'crossValidation': 10,
                    'parameters': { 'n_estimators': [5],
                                    'max_features': ['auto'],
                                    'random_state': [12345] }
                    }
"""


rf = RandomForest(argsRandomForest)
rf.findBestParameters()
rf.train()
rf.plotImportantParameters()
rf.predict()
rf.writeSubmissionFile()






argsLogisticRegression = {'trainData' : featureEngineering.trainData,
                          'testData': featureEngineering.testData,
                          'crossValidation': 10,
                          'parameters': { 'random_state': [12345],
                                          'penalty': ['l1', 'l2'],
                                          'C': [1, 5, 10, 50, 100, 500, 1000]}
                         }

lr = Logistic_Regression(argsLogisticRegression)
lr.findBestParameters()
lr.train()
lr.predict()
lr.writeSubmissionFile()






argsSupportVectorMachines = {'trainData' : featureEngineering.trainData,
                             'testData': featureEngineering.testData,
                             'random_state': 12345,
                             'crossValidation': 10,
                             'parameters': {#'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
                                            'kernel': ['rbf', 'poly'],
                                            'C': [1,5,10,50,100,500,1000],
                                            'random_state': [12345] }
                            }
svm = SupportVectorMachines(argsSupportVectorMachines)
svm.findBestParameters()
svm.train()
svm.predict()
svm.writeSubmissionFile()






argsNeuronalNet = {'trainData' : featureEngineering.trainData,
                             'testData': featureEngineering.testData,
                             'random_state': 12345,
                             'crossValidation': 10,
                             'parameters': {'solver': ['lbfgs'],
                                            'alpha': [1e-5],
                                            'hidden_layer_sizes': [(5, 5)],
                                            'random_state': [12345] }
                            }

nn = NeuronalNet(argsNeuronalNet)
nn.findBestParameters()
nn.train()
nn.predict()
nn.writeSubmissionFile()







argsVotingModels = {'trainData' : featureEngineering.trainData,
                    'testData': featureEngineering.testData,
                    'parameters': { 'estimators':[ ('lr', lr.fittedLogisticRegressionModel),
                                                   ('rf', rf.fittedRandomForestModel),
                                                   ('svm', svm.fittedSVCModel),
                                                   ('nn', nn.fittedMLPClassifierModel)], 'voting':'soft', 'weights':[1, 1, 1, 1] }
                   }

vm = VotingModels(argsVotingModels)
vm.train()
vm.predict()
vm.writeSubmissionFile()



print("--- %s seconds ---" % (time.time() - start_time))
