from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from train_and_test_definition import X_train
from fit_tune_function import fit_tune_store_sgdcv

len_n = len(X_train)

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)),
])

parameters = {
    'vect__ngram_range': [(1, 1), ],
    'tfidf__use_idf': (True, False),
    'clf__random_state': (0, 1, 50, 100, ),
    'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'clf__solver': ['lbfgs', 'sgd', 'adam'],
    'clf__hidden_layer_sizes': [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,)
    ],
    'clf__max_iter': (200, 400)
}

mlp_clf_gscv = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
fit_tune_store_sgdcv(mlp_clf_gscv, 'mlp')
