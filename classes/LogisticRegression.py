from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class Logistic_Regression:
    def __init__(self, args):
        self.args = args

        self.trainX = self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1)
        self.trainY = self.args['trainData']['Survived']

    def findBestParameters(self):
        grid_search = GridSearchCV(estimator=LogisticRegression(),
                                   param_grid=self.args['parameters'],
                                   cv=self.args['crossValidation'],
                                   scoring='accuracy')

        print("LR parameters:")
        # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        grid_search.fit(self.trainX, self.trainY)
        print("Best score: %0.3f" % grid_search.best_score_)

        self.bestScoreParamters = grid_search.best_estimator_.get_params()  # the dict of parameters with best score


    def train(self):
        self.fittedLogisticRegressionModel = LogisticRegression(**self.bestScoreParamters)
        self.fittedLogisticRegressionModel.fit(self.trainX, self.trainY)

    def predict(self):
        self.args['testData']['Survived'] = self.fittedLogisticRegressionModel.predict(self.args['testData'].drop(['Survived', 'PassengerId'], axis=1))

    def writeSubmissionFile(self):
        self.args['testData'].to_csv("data/Submission/Logistic/LRweakModel.csv", sep=',', index=False, columns = ['PassengerId', 'Survived'])