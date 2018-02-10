import numpy as np
import perceptron as per
import voted_perceptron as vper
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for l in range(0, 2):
    y = []
    X = []
    Y = []
    if l == 0:
        db_ = pd.read_csv("csv/page_blocks.csv")
        db_.columns = ['height', 'lenght', 'area', 'eccen', 'p_black',
                       'p_and', 'mean_tr', 'blackpix', 'blackand', 'wb_trans', 'val']
        y = db_[['val']]
        Y = [-1 if yi.val == 1 else 1 for yi in y.itertuples()]
        X = (db_.iloc[:, 0:10]).values
    else:
        db_ = pd.read_csv("csv/mushroom.csv")
        db_.columns = ['val', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                               'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                               'x21', 'x22', 'x23']

        y = db_[['val']]
        Y = [-1 if yi.val == 'e' else 1 for yi in y.itertuples()]
        X = (db_.iloc[:, 1:22]).values
        X = [[ord(c)/100 for c in xi] for xi in X]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train = np.insert(np.array(X_train), 0, 1, axis=1)
    X_test = np.insert(np.array(X_test), 0, 1, axis=1)

    if l == 0:
        print "****    PAGE BLOCKS     ****"
    else:
        print "****    MUSHROOM     ****"

    print "---------------------"
    print " STANDARD PERCEPTRON"

    obj_per = per.Perceptron(0.25, 500)

    res_standard, i_standard, lne = obj_per.training(X_train, y_train)
    print res_standard, i_standard, lne

    res_Y_per = [obj_per.predict(x) for x in X_test]

    cm = confusion_matrix(y_test, res_Y_per)
    np.set_printoptions(precision=3)
    print("Singol Layer accuracy: %.2f%%" % (100*accuracy_score(y_test, res_Y_per)))

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
    print(classification_report(y_test, res_Y_per))

    if l == 0:
        print "****    PAGE BLOCKS     ****"
    else:
        print "****    MUSHROOM     ****"

    print "--------------------"
    print "  VOTED PERCEPTRON"

    obj_vper = vper.voted_Perceptron(0.25, 1)

    res_voted, i_voted, lne = obj_vper.training(X_train, y_train)
    print res_voted, i_voted, lne

    res_Y_vper = [obj_vper.predict(x) for x in X_test]

    cmv = confusion_matrix(y_test, res_Y_vper)
    np.set_printoptions(precision=3)
    print("Singol Layer accuracy: %.2f%%" % (100*accuracy_score(y_test, res_Y_vper)))

    print

    # Plot non-normalized confusion matrix
    print 'Confusion matrix, without normalization'
    print cmv
    print
    print

    # Plot normalized confusion matrix
    print 'Normalized confusion matrix'
    print cmv.astype('float') / cmv.sum(axis=1)[:, np.newaxis]
    print
    print
    # print(classification_report(Yp_test, res_Y))
    print(classification_report(y_test, res_Y_vper))
