from sklearn import preprocessing
from sklearn import linear_model
import pandas as pd
import numpy as np

class FeatureEngineering:

    def __init__(self, args):
        self.initSettings(args)

        self.__transforamColumns()

        self.mergedOrigin = pd.concat([ self.trainData, self.testData ])

    def initSettings(self, args):
        self.originTrainData = args['originTrainData']
        self.originTestData = args['originTestData']

        self.trainData = args['originTrainData']
        self.testData = args['originTestData']

        self.featureLinearModel = args['featuresLinearModel']




    def linearPredictEmptyCells(self):
        self.linearPredictionAge({ 'Target': 'Age',
                                   'Features': self.featureLinearModel,
                                   'PredictName': 'Linear_Age'})

        self.linearPredictionAge({ 'Target': 'Fare',
                                   'Features': self.featureLinearModel,
                                   'PredictName': 'Linear_Fare'})





    def linearPredictionAge(self, args):
        target = args['Target']
        featuresLinearPredict = args['Features']
        predictName = args['PredictName']

        mergedOrigin = self.mergedOrigin
        # Drop all NaNs
        mergedOrigin = mergedOrigin[np.isfinite(mergedOrigin[target])]

        # Create linear regression object
        lineareRegressionModel = linear_model.LinearRegression()

        # Train the model using the training sets
        lineareRegressionModel.fit(mergedOrigin[featuresLinearPredict], mergedOrigin[target])

        # Create a new column Linear_Age
        self.trainData[predictName] = lineareRegressionModel.predict(self.trainData[featuresLinearPredict])
        self.testData[predictName] = lineareRegressionModel.predict(self.testData[featuresLinearPredict])


    """
    Combine the linear information with the original data (Age, Fare)
    """
    def combineNaNCellWithLinearPrediction(self):
        # Create a new column with the combination of original Age and linear prediction
        self.testData['AgeAndLinear'] = self.testData['Age']
        self.trainData['AgeAndLinear'] = self.trainData['Age']
        self.testData['AgeAndLinear'].fillna(self.testData['Linear_Age'], inplace=True)
        self.trainData['AgeAndLinear'].fillna(self.trainData['Linear_Age'], inplace=True)

        # Create a new column with the combination of original Fare and linear prediction
        self.testData['FareAndLinear'] = self.testData['Fare']
        self.trainData['FareAndLinear'] = self.trainData['Fare']
        self.testData['FareAndLinear'].fillna(self.testData['Linear_Fare'], inplace=True)
        self.trainData['FareAndLinear'].fillna(self.trainData['Linear_Fare'], inplace=True)




    #self.dataframe = dataframe
        #self.encoder = preprocessing.LabelEncoder()

    #def initFeatureEngineering(self):
    #    pass

    #def renameColumns(self, renameList):
    #    self.dataframe.columns = renameList

    def removeUnnecessaryColumns(self):
        del self.trainData['Cabin']
        del self.testData['Cabin']


    def __transforamColumns(self):
        encoder = preprocessing.LabelEncoder()

        # Sex
        self.trainData['Sex'] = encoder.fit_transform(self.trainData['Sex'])
        self.testData['Sex'] = encoder.fit_transform(self.testData['Sex'])

        # Embarked
        self.trainData['Embarked'] = encoder.fit_transform(self.trainData['Embarked'].fillna('S'))
        self.testData['Embarked'] = encoder.fit_transform(self.testData['Embarked'].fillna('S'))

        #self.dataframe['Embarked'] = encoder.fit_transform(self.testData['Embarked'].fillna('-1'))

        #self.dataframe['Sex.name'] = self.encoder.fit_transform(self.dataframe['Sex.name'].fillna('-1'))
        #dataset[i] = encoder.fit_transform(dataset[i].fillna('-1'))

    def extractInformations(self):
        self.__extractSex()

    def __extractSex(self):
        encoder = preprocessing.LabelEncoder()
        self.trainData['Male'] = encoder.fit_transform(self.trainData['Sex'] != 0)
        self.testData['Male'] = encoder.fit_transform(self.testData['Sex'] != 0)
        self.trainData['Female'] = encoder.fit_transform(self.trainData['Sex'] != 1)
        self.testData['Female'] = encoder.fit_transform(self.testData['Sex'] != 1)

    def createBins(self):
        self.__createBinAge()
        self.__createBinFare()

    """
    Create a bin für the linear combined ages
    """
    def __createBinAge(self):
        bins = [-5, 20, 40, 60, 80]
        group_names = [1, 2, 3, 4]
        self.testData['Bin_Age'] = pd.cut(self.testData['AgeAndLinear'], bins, labels=group_names)
        self.trainData['Bin_Age'] = pd.cut(self.trainData['AgeAndLinear'], bins, labels=group_names)

    """
    Create a bin für the linear combined ages
    """
    def __createBinFare(self):
        bins = [-10, 128, 256, 384, 513]
        group_names = [1, 2, 3, 4]
        self.testData['Bin_Fare'] = pd.cut(self.testData['FareAndLinear'], bins, labels=group_names)
        self.trainData['Bin_Fare'] = pd.cut(self.trainData['FareAndLinear'], bins, labels=group_names)


    def calculateFamily(self):
        print(self.trainData['SibSp'] != 0)

    def printNaNs(self):
        print("---------- Training Data ----------")
        print(self.trainData.isnull().sum())
        print("---------- Test Data ----------")
        print(self.testData.isnull().sum())
        print("-"*100)