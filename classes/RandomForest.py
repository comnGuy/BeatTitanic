import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing


class RandomForest:



    def __init__(self, args):
        self.args = args

        self.trainX = preprocessing.scale(self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1))

        #self.trainX = self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1)
        self.trainY = self.args['trainData']['Survived']
        self.columnNames = self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1).columns




    def findBestParameters(self):
        grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                                   param_grid=self.args['parameters'],
                                   cv=self.args['crossValidation'],
                                   scoring='accuracy')

        print("RF parameters:")
        # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        grid_search.fit(self.trainX, self.trainY)
        print("Best score: %0.3f" % grid_search.best_score_)

        self.bestScoreParamters = grid_search.best_estimator_.get_params()  # the dict of parameters with best score


    def train(self):
        self.fittedRandomForestModel = RandomForestClassifier(**self.bestScoreParamters)
        self.fittedRandomForestModel.fit(self.trainX, self.trainY)


    def plotImportantParameters(self):
        feature_important = pd.DataFrame(index=self.columnNames, data=self.fittedRandomForestModel.feature_importances_, columns=['importance'])
        feature_important = feature_important.sort_values(by=['importance'], ascending=True)
        feature_important.plot(kind='barh', stacked=True, color=['cornflowerblue'], grid=False, figsize=(8, 5))
        #plt.show()


    def predict(self):
        self.args['testData']['Survived'] = self.fittedRandomForestModel.predict(self.args['testData'].drop(['Survived', 'PassengerId'], axis=1))


    def writeSubmissionFile(self):
        self.args['testData'].to_csv("data/Submission/Forest/weakModel.csv", sep=',', index=False, columns = ['PassengerId', 'Survived'])



        """
        Generate the subsets
        """
        """def generateSubsets(self):
            for numberSubset in self.subsetSelection:
                # Creating a saving spot for the subsets in the dictonary
                self.subsetSelection[numberSubset]['subsets'] = []
                for lengthSubset in range(1, len(self.subsetSelection[numberSubset]['features'])):
                    combinationsOfFeatures = itertools.combinations(self.subsetSelection[numberSubset]['features'], lengthSubset)
                    for combinateFeature in combinationsOfFeatures:
                        # Save the subsets
                        self.subsetSelection[numberSubset]['subsets'].append(list(combinateFeature))


        def calcuateSubsetAccuracy(self):
            # Iterate over the given subset sets
            for numberSubset in self.subsetSelection:
                self.subsetSelection[numberSubset]['accuracy'] = pd.DataFrame(columns=['#Trees', 'Features', 'Accuracy'])
                filepath = self.accuracyPathRandomForest + self.subsetSelection[numberSubset]['filename']

                # Exist the file or should be the file loaded out of the settings?
                if not Path(filepath).is_file() or not self.subsetSelection[numberSubset]['loadFile']:
                    # Iterate of the generated subset
                    for subset in self.subsetSelection[numberSubset]['subsets']:
                        # Iterate over the number of trees
                        for numberTree in self.numberTrees:
                            # TODO function
                            model = RandomForestClassifier(n_estimators=numberTree, random_state=12345, n_jobs=-1)
                            model.fit(self.trainData[subset], self.trainData['Survived'])

                            # TODO function
                            self.combinedData['Predicted'] = model.predict(self.combinedData[subset])
                            accuracy = accuracy_score(self.combinedData['Predicted'], self.combinedData['Survived'])

                            joined = ' - '.join(subset)
                            self.subsetSelection[numberSubset]['accuracy'].loc[len(self.subsetSelection[numberSubset]['accuracy']) + 1] = [numberTree, joined, accuracy]

                            #print(numberSubset)
                            print(joined + '(' + str(numberTree) + '): ' + str(accuracy))

                    self.subsetSelection[numberSubset]['accuracy'].to_csv(filepath, sep=',')
                else: # Loads generated accuracy
                    self.subsetSelection[numberSubset]['accuracy'] = pd.read_csv(filepath)
    """



    """def getBestFittedModel(self):
        tmpBestModel = { 'Accuracy': 0.0, '#Trees': 0, 'Features': ''}
        for numberSubset in self.subsetSelection:
            #print(self.subsetSelection[numberSubset]['accuracy'].groupby(['Features', '#Trees'])['Accuracy'].max())
            index = self.subsetSelection[numberSubset]['accuracy']['Accuracy'].idxmax()
            tmpBestAccuracy = self.subsetSelection[numberSubset]['accuracy']['Accuracy'][index]
            if tmpBestAccuracy > tmpBestModel['Accuracy']:
                tmpBestModel['Accuracy'] = tmpBestAccuracy
                tmpBestModel['#Trees'] = self.subsetSelection[numberSubset]['accuracy']['#Trees'][index]
                tmpBestModel['Features'] = self.subsetSelection[numberSubset]['accuracy']['Features'][index].split(' - ')

        tmpBestModel['model'] = self.__getFittedModel(tmpBestModel['#Trees'], tmpBestModel['Features'])
        return tmpBestModel


    def __getFittedModel(self, numberTree, features):
        model = RandomForestClassifier(n_estimators=numberTree, random_state=12345, n_jobs=-1)
        model.fit(self.trainData[features], self.trainData['Survived'])
        return model

    def predictResults(self, args):
        tmpTestData = self.testData
        tmpTestData['Survived'] = args['model'].predict(tmpTestData[args['Features']])
        return tmpTestData[ ['PassengerId', 'Survived']]

    def writeSubmissionFile(self, args):
        args['trainWithResults'].to_csv(args['path'], sep=',', index=False)"""