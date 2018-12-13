
from __future__ import absolute_import

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score


class SVM_CLASSIFIER:
    def __init__(self, param_grid, classifier, out_model, dim_reducer=None):
        self.param_grid = param_grid
        self.classifier = classifier
        self.dim_reducer = dim_reducer
        self.out_model = out_model
        self.grid = None
        self.model = None
        self.trained_model = None

    def get_model(self):
        if self.dim_reducer is not None:
            self.model = make_pipeline(self.dim_reducer, self.classifier)
        else:
            self.model = make_pipeline(self.classifier)
        return

    def grid_search(self):
        self.grid = GridSearchCV(self.model, self.param_grid)
        return

    def prepare_model(self):
        self.get_model()
        self.grid_search()

    def train(self, Xtrain, ytrain):
        self.grid.fit(Xtrain, ytrain)
        print(self.grid.best_params_)
        self.trained_model = self.grid.best_estimator_
        return

    def test(self, Xtest, ytest, label_names):
        yfit = self.trained_model.predict(Xtest)
        print(classification_report(ytest, yfit,
                                    target_names=label_names))

        # mat = confusion_matrix(ytest, yfit)
        # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
        #             xticklabels=dataset.label_names,
        #             yticklabels=dataset.label_names)
        # plt.xlabel('true label')
        # plt.ylabel('predicted label')
        # plt.show()

        precision, recall, fscore, support = score(ytest, yfit)
        accuracy = accuracy_score(ytest, yfit)
        print('accuracy: ', accuracy)
        #
        # print('precision: {}'.format(precision))
        # print('recall: {}'.format(recall))
        # print('fscore: {}'.format(fscore))
        # print('support: {}'.format(support))
        average_precision = 0
        for p in precision:
            average_precision = average_precision + p / len(precision)
        return {'accuracy': accuracy, 'average_precision': average_precision, 'precision': precision, 'recall': recall,
                'fscore': fscore, 'support': support}

    def save(self):
        joblib.dump(self.trained_model, self.out_model)
        return

    def load(self):
        return joblib.load(self.out_model)
