import pickle
from test_functions import load_instances, test_classifier, test_with_examples

clf, vect, transform = load_instances('Random_forest')
# Get prediction
X_test = pickle.load(open('data_process/data_sets/x_test.pkl', 'rb'))
X_test_counts = vect.transform(X_test)
X_test_tfidf = transform.transform(X_test_counts)
test_predict = clf.predict(X_test_tfidf)

test_classifier(clf,test_predict)
test_with_examples('Random_forest',clf,vect,transform)
