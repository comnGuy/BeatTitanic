from classes.FeatureEngineering import FeatureEngineering
from classes.ExplorateDataFrame import ExplorateDataFrame
import pandas as pd

ed = ExplorateDataFrame()


# Read CSV
originTrainData = pd.read_csv('data/origin/train.csv')
originTestData = pd.read_csv('data/origin/test.csv')



# Feature Engineering
featureEngineering = FeatureEngineering({'originTrainData': originTrainData,
                                          'originTestData': originTestData,
                                          'featuresLinearModel': ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']})

# Remove Columns
featureEngineering.removeUnnecessaryColumns()
# Predict the empty cells
featureEngineering.linearPredictEmptyCells()
# Combine the linear information with the original data (Age, Fare)
featureEngineering.combineNaNCellWithLinearPrediction()
# Extract information out of the columns to generate new columns
featureEngineering.extractInformations()
# Create bins with various features
featureEngineering.createBins()
# Calcute family
featureEngineering.calculateFamily()
#Show sum of entries with NaNs
featureEngineering.printNaNs()

#print(featureEngineering.trainData)








# Random Forest
#rf = RandomForest(trainDataFeatureEngineering.dataframe, testDataFeatureEngineering.dataframe, subData)
#rf.generateAccurayForModels()
#rf.writeOverviewToCSV()

#features = ['Pclass', 'Sex', 'Age', 'Sex.name', 'Age_scale']

#rf.predictTestData(10, features)
#rf.mergePredictedSubmission()
#print(rf.getPredictedAccuracy())
#rf.writeSubmission('Pclass-Sex-Age-Sex.name-Age_scale_10.csv')

#print(rf.trainingData)
#ed.printColumns(df)

#fg.test()