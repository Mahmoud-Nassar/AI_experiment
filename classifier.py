import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import random
import numpy as np
from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold

K = 30
FOLDS_NUMBER = 5


class SimpleClassifier:
    def __init__(self, XTrain, yTrain, XEvaluation, yEvaluation):
        self.classifier = DecisionTreeClassifier()
        self.XTrain = XTrain
        self.yTrain = yTrain
        self.XEvaluation = XEvaluation
        self.yEvaluation = yEvaluation
        self.yPrediction = None

    def fit(self):
        self.classifier.fit(self.XTrain, self.yTrain)

    def predict(self, XEvaluation=None):
        if XEvaluation is None:
            XEvaluation = self.XEvaluation
        self.yPrediction = self.classifier.predict(XEvaluation)
        self.yPrediction = self.yPrediction.reshape(-1, 1)

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
                classifier.getPrecision()
                self.classifiers.append(classifier)

            XTest = self.X.iloc[testIndexes]
            yTest = self.y.iloc[testIndexes]

            for classifier in self.classifiers:
                classifier.predict(XTest)

            majorityPrediction = ClassifierCommittee.getModeArray(self.classifiers)
            majorityApproachPrecision += ClassifierCommittee.getPrecision(majorityPrediction, yTest)

            random21Classifiers = np.array(random.sample(self.classifiers, 21))
            random21Prediction = ClassifierCommittee.getModeArray(random21Classifiers)
            random21ApproachPrecision += ClassifierCommittee.getPrecision(random21Prediction, yTest)


        majorityApproachPrecision /=FOLDS_NUMBER
        random21ApproachPrecision /=FOLDS_NUMBER



    # """
    # this function recives an array of classifers and returns a
    # prediction array which represents the prediction of the majority
    # of the classifers supllied in the array self.classifiers
    # """
    # def getModeArray(self):
    #     classifiers = self.classifiers
    #     n = len(classifiers[0].yPrediction)
    #     result = np.zeros(n, dtype=int)
    #     for i in range(n):
    #         values = [c.yPrediction[i] for c in classifiers]
    #         value_counts = np.bincount(values)
    #         result[i] = np.argmax(value_counts)
    #     return result
    @staticmethod
    def getModeArray(classifiers):
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

    #
    # def getModeArray(self):
    #     classifiers = self.classifiers
    #     all_predictions = [classifier.yPrediction for classifier in classifiers]
    #     result = np.empty(len(all_predictions[0]), dtype=int)
    #     for i in range(len(result)):
    #         values = [pred[i] for pred in all_predictions if i < len(pred)]
    #         if len(values) > 0:
    #             counter = Counter(values)
    #             result[i] = counter.most_common(1)[0][0]
    #         else:
    #             result[i] = np.nan
    #     return result

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
