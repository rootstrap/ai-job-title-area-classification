from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from train_and_test_definition import X_train, y_train


clf_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',  alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])

clf_pipeline.fit(X_train, y_train)

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 3), (3, 4), ],
    'tfidf__use_idf': (True, False),
    'clf__random_state': (1, 21, 33, 42, 88, 100, 160),
    'clf__alpha': (1e-2, 1e-3, 1e-4, 0.1, 1e-6, ),
    'clf__max_iter': (2, 5, 10, 20, 100, 200)
}

gs_clf = GridSearchCV(clf_pipeline, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)


with open('best_params_sgd.py', 'w') as file:
    file.write(f"best_vect_ngram_range = {gs_clf.best_params_['vect__ngram_range']}\n")
    file.write(f"best_use_idf = {gs_clf.best_params_['tfidf__use_idf']}\n")
    file.write(f"best_alpha = {gs_clf.best_params_['clf__alpha']}\n")
    file.write(f"best_random_state = {gs_clf.best_params_['clf__random_state']}\n")
    file.write(f"best_max_iter = {gs_clf.best_params_['clf__max_iter']}\n")
    file.close()
