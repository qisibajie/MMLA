from sklearn import svm

# SVC and NuSVC implement the “one-against-one” approach
def oneAgainstOne():
    X = [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    Y = [0, 1, 2, 3]
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X, Y)
    c = clf.predict([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1]])
    class_type = clf.decision_function_shape
    print(c)
    print(class_type)

    dec = clf.decision_function([[0, 0, 0, 1]])
    print(dec)
    print(dec.shape[1])

def oneAgainstRest():
    X = [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    Y = [0, 1, 2, 3]
    clf = svm.LinearSVC()
    clf.fit(X, Y)
    c = clf.predict([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1]])
    print(c)

    dec = clf.decision_function([[0, 0, 0, 1]])
    print(dec)
    print(dec.shape[1])

# oneAgainstOne()
oneAgainstRest()