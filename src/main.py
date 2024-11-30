import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':

    # load the iris dataset. this should include all the data points and the labels
    # Note that this is a Supervised Learning, and we would be feeding the model with
    # both data (input) and labels (output)
    iris_dataset = load_iris()

    # it's a convention to have the input as X and output as y
    X = iris_dataset['data']
    y = iris_dataset['target']

    # X is a two-dimensional array where each row represents a data point and
    # each column represents a feature. And y is a one-dimensional array that just contains output for
    # each data point.
    # We can actually print the shape of X and y and see
    print('X shape: {}'.format(X.shape))
    print('y shape: {}'.format(y.shape))

    # next, we need to split the dataset into train set and test set
    # we can't feed the model with all the dataset otherwise it could just memorize rather than learning
    # patterns to handle prediction. so we split the dataset and train the model on the train set
    # after the model has been trained on patterns, we can feed it the test set to make predictions

    # here, we set the test_size to 25% so we get 25% of the dataset as our test set. and the random_state
    # is set to 42. The random_state value is just a seed, and it helps to always return the same datasets for
    # train set and test set. Not passing any value would mean that the set is always shuffled and random datasets
    # are returned
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # we use the K-Nearest Neighbors algorithm to classify our new data
    knn = KNeighborsClassifier(n_neighbors=1)

    # train the model of the train set
    knn.fit(X_train, y_train)

    # now make a prediction on the X_test, and then we can compare the prediction result with
    # the actual y_test (output) values
    y_pred = knn.predict(X_test)

    # we could actually manually test the accuracy of the prediction
    # When comparing predictions (y_pred) with ground truth labels (y_test), you perform an element-wise equality check:
    #
    # True (1) if the prediction matches the label.
    # False (0) if the prediction does not match the label.
    correctness = y_pred == y_test # this should return an array of boolean values

    print('Correctness array: {}'.format(correctness))

    # now we can check the mean of correctness. what this actually does is that it helps calculate the
    # proportion of correct predictions
    accuracy = np.mean(correctness)
    print('Accuracy using mean: {:.2f}'.format(accuracy))

    # or we could just ask the K-Nearest Neighbors algorithm to calculate the accuracy
    accuracy = knn.score(X_test, y_test)
    print('Accuracy score by knn: {:.2f}'.format(accuracy))

    print(f'From this, we can expect our model to be correct {(accuracy * 100)}% of the time for new irises')

    # finally let's allow the model to make a prediction on new unseen data
    # create the data point
    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction = knn.predict(X_new)

    print("Prediction: {}".format(prediction))
    print("Predicted target name: {}".format( iris_dataset['target_names'][prediction]))