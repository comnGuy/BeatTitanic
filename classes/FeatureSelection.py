import pandas as pd
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier



class FeatureSelection:
    def __init__(self, args):
        self.args = args

        self.trainX = self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1)
        self.trainY = self.args['trainData']['Survived']

        self.columnNames = self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1).columns

    #https://machinelearningmastery.com/feature-selection-machine-learning-python/
    def univerateSelection(self):
        # feature extraction
        test = SelectKBest(score_func=chi2, k=4)
        fit = test.fit(self.trainX, self.trainY)
        # summarize scores
        numpy.set_printoptions(precision=3)
        return fit.scores_


    def recursiveFeatureElimination(self):
        rfe = RFE(LogisticRegression(), 5)
        fit = rfe.fit(self.trainX, self.trainY)

        print("Num Features: " + str(fit.n_features_))
        print("Selected Features: " + str(fit.support_))
        print("Feature Ranking: " + str(fit.ranking_))
        return fit.ranking_


    def PCA(self):
        # feature extraction
        pca = PCA(n_components=3)
        fit = pca.fit(self.trainX)
        # summarize components
        print("Explained Variance: " + str(fit.explained_variance_ratio_))
        print(fit.components_)
        return fit.components_


    def importanceExtraTrees(self):
        model = ExtraTreesClassifier()
        model.fit(self.trainX, self.trainY)
        print(model.feature_importances_)
        return  model.feature_importances_


    def barPlot(self, fitScores, ascending = True, columns='importance', sort = True):
        feature_important = pd.DataFrame(index=self.columnNames, data=fitScores,
                                         columns=[columns])
        if sort:
            feature_important = feature_important.sort_values(by=[columns], ascending=ascending)
        feature_important.plot(kind='barh', stacked=True, color=['#FF8680'], grid=False, figsize=(8, 5))




    def showPlots(self):
        plt.show()