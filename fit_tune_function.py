import pickle

from train_and_test_definition import X_train, y_train


def fit_tune_store_sgdcv(sgd_clf, clf_type):
    print(f'Starting to fit and tune {clf_type.upper()} classifier...(it may take a while)')
    sgd_clf.fit(X_train, y_train)

    # Store the classifier in a .pkl file
    with open(f'{clf_type}_instances/{clf_type}_clf.pkl', 'wb') as file:
        pickle.dump(sgd_clf.best_estimator_['clf'], file)

    # Store the count vectorizer and tfidf transformer
    with open(f'{clf_type}_instances/{clf_type}_count_v.pkl', 'wb') as file:
        pickle.dump(sgd_clf.best_estimator_['vect'], file)

    with open(f'{clf_type}_instances/{clf_type}_tfidf_t.pkl', 'wb') as file:
        pickle.dump(sgd_clf.best_estimator_['tfidf'], file)

    print(f'{clf_type.upper()} fitted and tuned successfully!')
