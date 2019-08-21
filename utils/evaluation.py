import numpy as np


def fast_cm(tru, pred, num_classes):
    """
    Compute confusion matrix quickly using `np.bincount`
    :param tru: true class
    :param pred: predicted class
    :param num_classes: number of classes
    :return: confusion matrix
    """
    bin = tru * num_classes + pred
    h = np.bincount(bin, minlength=num_classes*num_classes)
    return h.reshape((num_classes, num_classes))



class EvaluatorIoU (object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes))

    def sample(self, truth, prediction, ignore_value=None):
        truth = truth.flatten()
        prediction = prediction.flatten()

        if ignore_value is not None:
            mask = truth != ignore_value
            truth = truth[mask]
            prediction = prediction[mask]

        self.cm += fast_cm(truth, prediction, self.num_classes)

    @property
    def intersection(self):
        return np.diag(self.cm)

    @property
    def union(self):
        return self.cm.sum(axis=0) + self.cm.sum(axis=1) - np.diag(self.cm)

    def score(self):
        return self.intersection.astype(float) / np.maximum(self.union.astype(float), 1.0)

