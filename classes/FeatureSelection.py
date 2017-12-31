import pandas as pd
import numpy as np, numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.cross_validation import KFold



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

    def k_fold_to_get_acc(self, clf, x_split, y_split):
        kf = KFold(len(y_split), n_folds=10)
        acc_t = 0
        acc_v = 0
        for train, valid in kf:
            x_train, y_train = x_split.iloc[train], y_split.iloc[train]
            x_valid, y_valid = x_split.iloc[valid], y_split.iloc[valid]

            clf = clf.fit(x_train, y_train)
            pred = clf.predict(x_train)
            acc_t += np.mean(pred == y_train)  # accuracy on train
            pred = clf.predict(x_valid)
            acc_v += np.mean(pred == y_valid)  # accuracy on validation
        return acc_t * 10, acc_v * 10

    def ablative_analysis(self, clf, train, feats):
        x, y = train.drop('Survived', 1), train["Survived"]
        acc = {}
        dropped = []
        print('all reserved - %%%.2f  %%%.2f' % self.k_fold_to_get_acc(clf, x, y))
        while (len(feats) > 0):
            item = feats[0]
            feats = feats[1:]
            dropped.append(item)
            acc[item] = self.k_fold_to_get_acc(clf, x.drop(dropped, 1), y)
        for key in acc:
            acc_t, acc_v = acc[key]
            print('del %s -> %%%.2f  %%%.2f' % (key, acc_t, acc_v))
        print('Remains:')
        print(list(x.drop(dropped, 1).columns))


    def showPlots(self):
        plt.show()