from sklearn.ensemble import VotingClassifier
import pandas as pd


class VotingModels:

    def __init__(self, args):
        self.args = args

        self.trainX = self.args['trainData'].drop(['Survived', 'PassengerId'], axis=1)
        self.trainY = self.args['trainData']['Survived']

    #def voting(self):

    #    results = eclf1.predict(self.args['testData'])
    #    output = pd.DataFrame({'PassengerId': id_list, "Survived": results})
    #    output.to_csv('prediction.csv', index=False)

    def train(self):
        self.fittedVotingClassifier = VotingClassifier(estimators   =   self.args['parameters']['estimators'],
                                                       voting       =   self.args['parameters']['voting'],
                                                       weights      =   self.args['parameters']['weights'])
        self.fittedVotingClassifier = self.fittedVotingClassifier.fit(self.trainX, self.trainY)

    def predict(self):
        self.args['testData']['Survived'] = self.fittedVotingClassifier.predict(self.args['testData'].drop(['Survived', 'PassengerId'], axis=1))

    def writeSubmissionFile(self):
        self.args['testData'].to_csv("data/Submission/Voting/0.csv", sep=',', index=False, columns = ['PassengerId', 'Survived'])