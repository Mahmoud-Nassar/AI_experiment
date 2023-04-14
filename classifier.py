import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from helperFunctionsAndVariables.globalVariables import \
    csvProcessedDataReadPath, attributes, classificationField, \
    generalizationFactor, kFoldNumSplits, weightMap
from helperFunctionsAndVariables.helperFunctions import \
    createGraph, createMultipleFunctionGraph, createMultipleFunctionTable

K = 101


class SimpleClassifier:
    def __init__(self, XTrain, yTrain, XEvaluation,yEvaluation):
        self.classifier = DTClassifier()
        self.XTrain = XTrain
        self.yTrain = yTrain
        self.XEvaluation = XEvaluation
        self.yEvaluation = yEvaluation

    def train(self):
        self.Classifier.fit(self.XTrain, self.yTrain)

    def predict(self):
        self.y_pred = self.Classifier.predict(self.X_test)
        self.y_pred = self.y_pred.reshape(-1, 1)




class DTClassifier:
    def __init__(self, csvDataReadPath, attributes, classificationField):
        self.df = pd.read_csv(csvDataReadPath)
        self.X = df[attributes]
        self.y = df[classificationField]
        self.kf = KFold(n_splits=5, shuffle=True, random_state=15161098)
        self.folds = []
        for train_indexes, test_indexes in self.kf.split(X):
            self.folds.append((train_indexes, test_indexes))

        self.Classifiers = None

    def majorityExperiment(self):
        for train_indexes, test_indexes in self.folds:
            XTrain = X.iloc[train_indexes]
            yTrain = y.iloc[train_indexes]

            for i in K:
                innerTrainIndexes, innerEvaluationIndexes = \
                    train_test_split(train_indexes, test_size=0.33)

                XInnerTrain = X.iloc[innerTrainIndexes]
                yInnerTrain = y.iloc[innerTrainIndexes]

                XInnerEvaluation = X.iloc[innerEvaluationIndexes]
                yInnerEvaluation = y.iloc[innerEvaluationIndexes]

                classifier = DecisionTreeClassifier()
                classifier.fit(XInnerTrain, yInnerTrain)

                classifier = DTClassifier()
                classifier.train()
                classifier.predict()
                precisionSum += classifier.getPrecision()
            precision = precisionSum / kFoldNumSplits

            X_test = X.iloc[test_indexes]
            y_test = y.iloc[test_indexes]
