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
        self.originGenderSubmission = args['originGenderSubmission']

        self.testData['Survived'] = self.originGenderSubmission['Survived']

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


    def normalize(self):
        normalizeColumns = { 'AgeAndLinearNormalize': 'AgeAndLinear',
                             'FareAndLinearNormalize': 'FareAndLinear',
                             'LinearAgeNormalize': 'Linear_Age',
                             'LinearFareNormalize': 'Linear_Fare'}

        for newColumn in normalizeColumns:
            self.testData[newColumn] = (self.testData[normalizeColumns[newColumn]] - self.testData[normalizeColumns[newColumn]].min()) / \
                                                     (self.testData[normalizeColumns[newColumn]].max() - self.testData[normalizeColumns[newColumn]].min())

            self.trainData[newColumn] = (self.trainData[normalizeColumns[newColumn]] - self.trainData[normalizeColumns[newColumn]].min()) / \
                                                      (self.trainData[normalizeColumns[newColumn]].max() - self.trainData[normalizeColumns[newColumn]].min())


    def removeUnnecessaryColumns(self):
        del self.trainData['Cabin']
        del self.testData['Cabin']

        del self.trainData['Age']
        del self.testData['Age']

        del self.trainData['Fare']
        del self.testData['Fare']

        del self.trainData['Name']
        del self.testData['Name']

        del self.trainData['Ticket']
        del self.testData['Ticket']

        # Delete for Feature Selections, cause there are negative values
        del self.trainData['AgeAndLinear']
        del self.testData['AgeAndLinear']
        del self.trainData['FareAndLinear']
        del self.testData['FareAndLinear']
        del self.trainData['Linear_Age']
        del self.testData['Linear_Age']
        del self.trainData['Linear_Fare']
        del self.testData['Linear_Fare']

    def removeVotedFeatures(self):
        del self.trainData['Bin_Age']
        del self.testData['Bin_Age']
        del self.trainData['Bin_Fare']
        del self.testData['Bin_Fare']
        del self.trainData['Parch']
        del self.testData['Parch']
        del self.trainData['SibSp']
        del self.testData['SibSp']
        del self.trainData['Family']
        del self.testData['Family']
        del self.trainData['#Family']
        del self.testData['#Family']


    def __transforamColumns(self):
        encoder = preprocessing.LabelEncoder()

        # Sex
        self.trainData['Sex'] = encoder.fit_transform(self.trainData['Sex'])
        self.testData['Sex'] = encoder.fit_transform(self.testData['Sex'])

        # Embarked
        self.trainData['Embarked'] = encoder.fit_transform(self.trainData['Embarked'].fillna('S'))
        self.testData['Embarked'] = encoder.fit_transform(self.testData['Embarked'].fillna('S'))


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
        self.trainData['Bin_Fare'] = pd.cut(self.trainData['FareAndLinear'], bins, labels=group_names)
        self.testData['Bin_Fare'] = pd.cut(self.testData['FareAndLinear'], bins, labels=group_names)


    def extractPersonTitle(self):
        encoder = preprocessing.LabelEncoder()
        self.trainData['PersonTitle'] = encoder.fit_transform(self.trainData['Name'].str.split(', ').str[1].str.split('\.').str[0])
        self.testData['PersonTitle'] = encoder.fit_transform(self.testData['Name'].str.split(', ').str[1].str.split('\.').str[0])


    """
    Extract the information of the family
    """
    def calculateFamily(self):
        # Family true|false
        self.trainData['Family'] = np.logical_or(self.trainData['SibSp'] != 0,
                                                 self.trainData['Parch'] != 0)
        self.testData['Family'] = np.logical_or(self.testData['SibSp'] != 0,
                                                self.testData['Parch'] != 0)
        # Number of Family
        self.trainData['#Family'] = self.trainData['SibSp'] + self.trainData['Parch']
        self.testData['#Family'] = self.testData['SibSp'] + self.testData['Parch']


    def printNaNs(self):
        print("---------- Training Data ----------")
        print(self.trainData.isnull().sum())
        print("---------- Test Data ----------")
        print(self.testData.isnull().sum())
        print("-"*100)