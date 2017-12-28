from sklearn.linear_model import LinearRegression, LogisticRegression


class Logistic_Regression:
    def __init__(self, args):
        self.args = args

        self.trainX = self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1)
        self.trainY = self.args['trainData']['Survived']

    def train(self):
        self.fittedLogisticRegressionModel = LogisticRegression(random_state=self.args['random_state'])
        self.fittedLogisticRegressionModel.fit(self.trainX, self.trainY)

    def predict(self):
        self.args['testData']['Survived'] = self.fittedLogisticRegressionModel.predict(self.args['testData'].drop(['Survived', 'PassengerId'], axis=1))

    def writeSubmissionFile(self):
        self.args['testData'].to_csv("data/Submission/Logistic/LRweakModel.csv", sep=',', index=False, columns = ['PassengerId', 'Survived'])