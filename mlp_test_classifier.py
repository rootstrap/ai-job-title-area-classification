import pickle
from test_functions import load_instances, test_classifier, test_with_examples


clf, count_vect, tfidf_transformer = load_instances('mlp')
X_test = pickle.load(open('data_process/data_sets/x_test.pkl', 'rb'))

# Test and show results
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
test_predict = clf.predict(X_test_tfidf)
test_classifier(clf, test_predict)
test_with_examples('mlp', clf, count_vect, tfidf_transformer)
