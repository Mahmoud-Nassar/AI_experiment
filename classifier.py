import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import random
import numpy as np
import heapq
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold

K = 101
FOLDS_NUMBER = 5
COMMITTEE_MEMBERS_NUMBER = 21


class SimpleClassifier:
    def __init__(self, XTrain, yTrain, XEvaluation, yEvaluation):
        self.classifier = DecisionTreeClassifier()
        self.XTrain = XTrain
        self.yTrain = yTrain
        self.XEvaluation = XEvaluation
        self.yEvaluation = yEvaluation
        self.yPrediction = None
        self.evaluationPrecision = None

    def fit(self):
        self.classifier.fit(self.XTrain, self.yTrain)

    def predict(self, XEvaluation=None):
        if XEvaluation is None:
            XEvaluation = self.XEvaluation
        self.yPrediction = self.classifier.predict(XEvaluation)
        self.yPrediction = self.yPrediction.reshape(-1, 1)

    def saveEvaluationPrecision(self):
        self.evaluationPrecision = self.getPrecision()

    def getPrecision(self):
        i = 0
        wrong = 0
        for (index, row) in self.yEvaluation.iterrows():
            if self.yPrediction[i] != int(row.values):
                wrong += 1
            i += 1
        precision = (len(self.yPrediction) - wrong) / len(self.yPrediction)
        return precision * 100


class ClassifierCommittee:
    def __init__(self, csvDataReadPath, attributes, classificationField):
        # df = pd.read_csv(csvDataReadPath, nrows=200)
        df = pd.read_csv(csvDataReadPath)
        self.X = df[attributes]
        self.y = df[classificationField]
        # Fill missing values with the mean of each column
        for column in self.X.columns:
            self.X.loc[self.X[column].isnull(), column] = self.X[column].mean()
        # # delete the lines with missing values
        # self.X.dropna(subset=self.X.columns, inplace=True)
        # self.y = self.y[self.X.index]

        self.kf = KFold(n_splits=FOLDS_NUMBER, shuffle=True, random_state=15161098)
        self.folds = []
        for trainIndexes, testIndexes in self.kf.split(self.X):
            self.folds.append((trainIndexes, testIndexes))
        self.classifiers = []

    def experiment(self):
        majorityApproachPrecision = 0
        random21ApproachPrecision = 0
        bestPrecision21Precision = 0
        distanceBasedPrecsison = 0
        for trainIndexes, testIndexes in self.folds:
            XTrain = self.X.iloc[trainIndexes]
            yTrain = self.y.iloc[trainIndexes]

            for i in range(0, K):
                innerTrainIndexes, innerEvaluationIndexes = \
                    train_test_split(trainIndexes, test_size=0.33)

                XInnerTrain = self.X.iloc[innerTrainIndexes]
                yInnerTrain = self.y.iloc[innerTrainIndexes]
                XInnerEvaluation = self.X.iloc[innerEvaluationIndexes]
                yInnerEvaluation = self.y.iloc[innerEvaluationIndexes]

                classifier = SimpleClassifier(XInnerTrain, yInnerTrain
                                              , XInnerEvaluation, yInnerEvaluation)
                classifier.fit()
                classifier.predict()
                classifier.saveEvaluationPrecision()
                self.classifiers.append(classifier)

            XTest = self.X.iloc[testIndexes]
            yTest = self.y.iloc[testIndexes]

            for classifier in self.classifiers:
                classifier.predict(XTest)

            majorityPrediction = ClassifierCommittee.getModeArray(self.classifiers)
            majorityApproachPrecision += ClassifierCommittee.getPrecision(majorityPrediction, yTest)

            random21Classifiers = np.array(random.sample(self.classifiers, COMMITTEE_MEMBERS_NUMBER))
            random21Prediction = ClassifierCommittee.getModeArray(random21Classifiers)
            random21ApproachPrecision += ClassifierCommittee.getPrecision(random21Prediction, yTest)

            bestPrecision21Classifiers = self.getBestAccuracyClassifiers(COMMITTEE_MEMBERS_NUMBER)
            bestPrecision21Prediction = ClassifierCommittee.getModeArray(bestPrecision21Classifiers)
            bestPrecision21Precision += ClassifierCommittee.getPrecision(bestPrecision21Prediction, yTest)

            distanceBasedPrecsison += self.getDistanceBasedPrecsison(XTest, yTest)

        majorityApproachPrecision /= FOLDS_NUMBER
        random21ApproachPrecision /= FOLDS_NUMBER
        bestPrecision21Precision /= FOLDS_NUMBER
        distanceBasedPrecsison /= FOLDS_NUMBER

        return majorityApproachPrecision, random21ApproachPrecision, \
            bestPrecision21Precision, distanceBasedPrecsison

    def getDistanceBasedPrecsison(self, XTest, yTest):
        right = 0
        i = 0
        for index, example in XTest.iterrows():
            distances = []
            for classifier in self.classifiers:
                distanceFromExample = \
                    ClassifierCommittee.distanceFromSet(classifier.XTrain, example)
                distances.append([classifier, distanceFromExample])
            distanceBest21Tuples = heapq.nsmallest(COMMITTEE_MEMBERS_NUMBER, distances,
                                                   key=lambda x: x[1])

            distanceBest21Classifiers = [node[0] for node in distanceBest21Tuples]
            distanceBest21Prediction = ClassifierCommittee.getModeArray(distanceBest21Classifiers)
            prediction = distanceBest21Prediction[i]
            realValue = yTest.loc[index][0]
            right += 1 if prediction == realValue else 0
            i += 1
        return (right / len(yTest) * 100)

    @staticmethod
    def distanceFromSet(XTrainSet, testExample):
        distance = 0
        for index, trainExample in XTrainSet.iterrows():
            distance += np.sqrt(((trainExample - testExample) ** 2).sum())
        distance /= len(XTrainSet)
        return distance

    @staticmethod
    def getModeArray(classifiers):
        """

        :param classifiers: an array of classifiers from type simpleClassifiers
        :return: the prediction as if it was deciding based on majority desicion
        """
        n_arrays = len(classifiers)
        array_length = len(classifiers[0].yPrediction)
        combined_array = np.zeros(array_length)

        for i in range(array_length):
            values = [classifier.yPrediction[i] for classifier in classifiers]
            value_counts = {}
            for value_array in values:
                value = value_array[0]
                if value not in value_counts:
                    value_counts[value] = 1
                else:
                    value_counts[value] += 1
            most_common_value = max(value_counts, key=value_counts.get)
            combined_array[i] = most_common_value
        return combined_array

    @staticmethod
    def getPrecision(yPrediction, yEvaluation):
        i = 0
        wrong = 0
        for (index, row) in yEvaluation.iterrows():
            if yPrediction[i] != int(row.values):
                wrong += 1
            i += 1
        precision = (len(yPrediction) - wrong) / len(yPrediction)
        return precision * 100

    def getBestAccuracyClassifiers(self, committeeSize):
        classifiers = self.classifiers
        sorted_classifiers = sorted(classifiers, key=lambda x: x.evaluationPrecision, reverse=True)
        return sorted_classifiers[:committeeSize]
