class Classifier:
    def __init__(self, labels, X, y):
        self.labels = labels
        self.X = X
        self.y = y


c1 = Classifier(['normal', 'covid', 'pneumonia'], [], [])
print(c1.labels)
