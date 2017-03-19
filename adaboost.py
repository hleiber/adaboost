import numpy as np
import argparse
from sklearn.tree import DecisionTreeClassifier

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    parser.add_argument('--numTrees', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def adaboost(X, y, num_iter):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is in {-1, 1}^n
    """
    trees = []
    trees_weights = []
    N = len(X)
    weights = np.ones(N)/N
    
    for n in range(num_iter):
        h = DecisionTreeClassifier(max_depth=1)
        h_fit = h.fit(X, y, sample_weight = weights)
        trees.append(h_fit)
        pred = h.predict(X)
        pred_error = (pred == y)
        top_error = 0
        for i in range(N):
            if pred_error[i] == False:
                top_error += weights[i]
        error = top_error / sum(weights)   
        alpha = np.log((1-error) / error)
        trees_weights.append(alpha)
        for i in range(N):
            if pred_error[i] == False:
                weights[i] = weights[i] * np.exp(alpha)
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y

    assume Y in {-1, 1}^n
    """
    pred_tree = []
    for h in trees:
        pred_tree.append(np.array(h.predict(X)))
    pred_tree = np.array(pred_tree)   
    trees_weights = np.array(trees_weights)
    gm_sum = np.sum((trees_weights*pred_tree.T).T, axis=0)
    Yhat = np.sign(gm_sum)
    return Yhat


def parse_spambase_data(filename): #done
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row.
    """
    arr = np.loadtxt(filename, delimiter=',')
    X = np.delete(arr,-1,1)
    Y = arr.T[-1,:]
    return X, Y

def new_label(Y):
    """ Transforms a vector od 0s and 1s in -1s and 1s.
    """
    return [-1. if y == 0. else 1. for y in Y]

def old_label(Y):
    return [0. if y == -1. else 1. for y in Y]

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y)) 

def main():
    """
    This code is called from the command line via
    
    python adaboost.py --train [path to filename] --test [path to filename] --numTrees 
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    num_trees = int(args['numTrees'][0])
    # training
    X, Y = parse_spambase_data(train_file)
    Y_new_label = new_label(Y)
    trees, trees_weights = adaboost(X, Y_new_label, num_trees)
    Yhat = adaboost_predict(X, trees, trees_weights)
    Yhat = old_label(Yhat)
    # testing
    X_test, Y_test = parse_spambase_data(test_file)
    Yhat_test = adaboost_predict(X_test, trees, trees_weights)
    Yhat_test = old_label(Yhat_test)
    
    full = np.loadtxt(test_file, delimiter=',')
    mat = np.matrix(full)
    y = np.transpose(np.matrix(Yhat_test))
    pred_df = np.append(mat, y, 1)
    np.savetxt('predictions.txt', pred_df, delimiter=',', fmt='%.1f')
    acc_test = accuracy(Y_test, Yhat_test)
    acc = accuracy(Y, Yhat)
    print("Train Accuracy %.4f" % acc)
    print("Test Accuracy %.4f" % acc_test)

if __name__ == '__main__':
    main()

