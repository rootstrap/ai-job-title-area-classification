from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from fit_tune_function import fit_tune_store_sgdcv


clf_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',  alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 3), (3, 4), ],
    'tfidf__use_idf': (True, False),
    'clf__random_state': (1, 21, 33, 42, 88, 100, 160),
    'clf__alpha': (1e-2, 1e-3, 1e-4, 0.1, 1e-6, ),
    'clf__max_iter': (2, 5, 10, 20, 100, 200)
}

sgd_clf_gscv = GridSearchCV(clf_pipeline, parameters, cv=5, iid=False, n_jobs=-1)
fit_tune_store_sgdcv(sgd_clf_gscv, 'sgd')
