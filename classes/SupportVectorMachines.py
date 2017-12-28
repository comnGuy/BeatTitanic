from sklearn import svm
from sklearn.model_selection import GridSearchCV

class SupportVectorMachines:
    def __init__(self, args):
        self.args = args

        self.trainX = self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1)
        self.trainY = self.args['trainData']['Survived']

    def findBestParameters(self):
        grid_search = GridSearchCV(estimator=svm.SVC(),
                                   param_grid=self.args['parameters'],
                                   cv=self.args['crossValidation'],
                                   scoring='accuracy')

        print("SVM parameters:")
        # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        grid_search.fit(self.trainX, self.trainY)
        print("Best score: %0.3f" % grid_search.best_score_)

        self.bestScoreParamters = grid_search.best_estimator_.get_params()  # the dict of parameters with best score
        self.bestScoreParamters['probability'] = True

    def train(self):
        self.fittedSVCModel = svm.SVC(**self.bestScoreParamters)
        self.fittedSVCModel.fit(self.trainX, self.trainY)

    def predict(self):
        self.args['testData']['Survived'] = self.fittedSVCModel.predict(self.args['testData'].drop(['Survived', 'PassengerId'], axis=1))

    def writeSubmissionFile(self):
        self.args['testData'].to_csv("data/Submission/SVM/SVMBetterModel.csv", sep=',', index=False, columns = ['PassengerId', 'Survived'])