from typing import List

class Scorer():

    def __init__(self,true_label,predicted_label):
        self.true_label = true_label
        self.predicted = predicted_label
        self.accuracy_score = None
        self.precition_score = None
        self.recall_score = None
        self.fp_rate = None
        self.fn_rate = None

    def get_confusion_matrix(self) -> List[List[int]]:
        """Computes confusion matrix from lists of actual or predicted labels.

        Args:
            actual: List of integers (0 or 1) representing the actual classes of
                some instances.
            predicted: List of integers (0 or 1) representing the predicted classes
                of the corresponding instances.

        Returns:
            List of two lists of length 2 each, representing the confusion matrix.
        """

        if len(self.true_label) != len(self.predicted):
            return "The predicted and the actual list must be of the same length, and their elements must correspond,"

        c_matrix = [[0,0],[0,0]]

        for i in range(len(self.true_label)):
            c_matrix[self.true_label[i]][self.predicted[i]] += 1


        return c_matrix


    def accuracy(self) -> None:
        """Computes the accuracy from lists of actual or predicted labels.

        Args:
            actual: List of integers (0 or 1) representing the actual classes of
                some instances.
            predicted: List of integers (0 or 1) representing the predicted classes
                of the corresponding instances.

        Returns:
            Accuracy as a float.
        """
        matrix = self.get_confusion_matrix()
        self.accuracy_score = (matrix[0][0]+matrix[1][1])/(sum(matrix[0]+matrix[1]))
        
    def precision(self) -> None:
        """Computes the precision from lists of actual or predicted labels.

        Args:
            actual: List of integers (0 or 1) representing the actual classes of
                some instances.
            predicted: List of integers (0 or 1) representing the predicted classes
                of the corresponding instances.

        Returns:
            Precision as a float.
        """
        matrix = self.get_confusion_matrix()
        self.precition_score = (matrix[1][1])/(matrix[1][1]+matrix[0][1])

    def recall(self) -> float:
        """Computes the recall from lists of actual or predicted labels.

        Args:
            actual: List of integers (0 or 1) representing the actual classes of
                some instances.
            predicted: List of integers (0 or 1) representing the predicted classes
                of the corresponding instances.

        Returns:
            Recall as a float.
        """
        matrix = self.get_confusion_matrix()
        self.recall_score = (matrix[1][1])/(matrix[1][1]+matrix[1][0])
        


    def f1(self) -> None:
        """Computes the F1-score from lists of actual or predicted labels.

        Args:
            actual: List of integers (0 or 1) representing the actual classes of
                some instances.
            predicted: List of integers (0 or 1) representing the predicted classes
                of the corresponding instances.

        Returns:
            float of harmonic mean of precision and recall.
        """
        self.f1_score = 2*((self.precition_score*self.recall_score)/(self.precition_score+self.recall_score))


    def false_positive_rate(self) -> None:
        """Computes the false positive rate from lists of actual or predicted
            labels.

        Args:
            actual: List of integers (0 or 1) representing the actual classes of
                some instances.
            predicted: List of integers (0 or 1) representing the predicted classes
                of the corresponding instances.

        Returns:
            float of number of instances incorrectly classified as positive divided
                by number of actually negative instances.
        """
        matrix = self.get_confusion_matrix()
        
        self.fp_rate = (matrix[0][1])/(matrix[0][1]+matrix[0][0]) 

    def false_negative_rate(self) -> None:
        """Computes the false negative rate from lists of actual or predicted
            labels.

        Args:
            actual: List of integers (0 or 1) representing the actual classes of
                some instances.
            predicted: List of integers (0 or 1) representing the predicted classes
                of the corresponding instances.

        Returns:
            float of number of instances incorrectly classified as negative divided
                by number of actually positive instances.
        """
        matrix = self.get_confusion_matrix()
        self.fn_rate = (matrix[1][0])/(matrix[1][0]+matrix[1][1])

    def calculate_scores(self):
        self.accuracy()
        self.precision()
        self.recall()
        self.f1()
        self.false_negative_rate()
        self.false_positive_rate()

    def print_all(self):
        self.calculate_scores()
        print(f'Accuracy: {self.accuracy_score}')
        print(f'Precition: {self.precition_score}')
        print(f'Recall: {self.recall_score}')
        print(f'f1: {self.f1_score}')
        print(f'False positive: {self.fp_rate}')
        print(f'False Negative: {self.fn_rate}')
