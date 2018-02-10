import seaborn as sns
import matplotlib.pyplot as plt
import perceptron as per
import voted_perceptron as v_per
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

#print "****     BANANA     ****"
db_ = pd.read_csv("csv/banana.csv")
db_.columns = ['val', 'x', 'y']
y = db_[['val']].values
Y = [-1 if yi == -1 else 1 for yi in y]
# print list(Y).count(-1), list(Y).count(1)
X = db_.iloc[:, 1:3]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

val = True

X_train = np.array(X_train.values)
X_test = np.array(X_test.values)


for l in range(0, 2):
    print "****     BANANA     ****"
    if val:
        print "---------------------"
        print " STANDARD PERCEPTRON"
    else:
        print "--------------------"
        print "  VOTED PERCEPTRON"

    X_train = np.insert(np.array(X_train), 0, 1, axis=1)
    X_test = np.insert(np.array(X_test), 0, 1, axis=1)

    obj = (per.Perceptron(0.25, 200) if val else v_per.voted_Perceptron(0.25, 1))

    res, i, k = obj.training(X_train, y_train)
    print res, i, k

    res_Y = [obj.predict(x) for x in X_test]

    X_train = np.delete(np.array(X_train), 0, axis=1)
    X_test = np.delete(np.array(X_test), 0, axis=1)

    plt.figure(l)
    plt.subplot(211)
    # plt.ylim(-0.5, 3)
    # plt.xlim(-0.5, 7)
    plt.scatter([x[0] for x in X_train], [x[1] for x in X_train], c=['g' if y is 1 else 'r' for y in y_train])
    if res[1] == 0:
        x1 = [1, -res[0]/res[2]]
    else:
        x1 = [-res[0]/res[1], 0]
    if res[2] == 0:
        x2 = [-res[0]/res[1], 1]
    else:
        x2 = [0, -res[0]/res[2]]
    if not(val):
        None
    else:
        newline(x1, x2)

    plt.subplot(212)
    # plt.ylim(-0.5, 3)
    # plt.xlim(-0.5, 7)
    plt.scatter([x[0] for x in X_train], [x[1] for x in X_train], c=['g' if y is 1 else 'r' for y in y_train])
    # plt.scatter([x[0] for x in X_test], [x[1] for x in X_test], marker='x', c=['g' if y is 1 else 'r' for y in res_Y])

    if not(val):
        None
    else:
        newline(x1, x2)
    plt.scatter([x[0] for x in X_test],
                [x[1] for x in X_test],
                marker='x',
                c=['b' if y is 1 else 'y' for y in res_Y])


    # Compute confusion matrix
    #   without plotting (for plot see how:
    #   http://scikit-learn.org/stable/auto_examples/
    #       model_selection/plot_confusion_matrix.html

    # cm = confusion_matrix(Yp_test, res_Y)
    cm = confusion_matrix(y_test, res_Y)
    np.set_printoptions(precision=3)

    # print("Singol Layer accuracy: %.2f%%" % (100*accuracy_score(Yp_test, res_Y)))
    print("Singol Layer accuracy: %.2f%%" % (100*accuracy_score(y_test, res_Y)))

    print

    # Plot non-normalized confusion matrix
    print 'Confusion matrix, without normalization'
    print cm
    print
    print

    # Plot normalized confusion matrix
    print 'Normalized confusion matrix'
    print cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print
    print
    # print(classification_report(Yp_test, res_Y))
    print(classification_report(y_test, res_Y))

    val = False

plt.show()