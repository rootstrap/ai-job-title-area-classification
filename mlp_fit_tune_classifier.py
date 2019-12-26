import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from fit_tune_function import fit_tune_store_sgdcv

X_train = pickle.load(open('data_process/data_sets/x_train.pkl', 'rb'))

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer(use_idf=False).fit_transform(X_train_counts)
len_n = tf_transformer.shape[1]

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MLPClassifier()),
])

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), ],
    'tfidf__use_idf': (True, False),
    'clf__random_state': (0, ),
    'clf__alpha': (1e-2, 1e-3, 1e-4, 0.1, 1e-5, ),
    'clf__hidden_layer_sizes': [(int(len_n/2), ), (int((len_n + 4)*(2/3)), ), (int(len_n/2), int(len_n/4))],
    'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'clf__solver': ['sgd', 'adam', 'lbfgs'],
    'clf__learning_rate': ['constant', 'adaptive'],
    'clf__validation_fraction': [0.1, 0.3, 0.5, 0.01]
}

mlp_clf_gscv = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
fit_tune_store_sgdcv(mlp_clf_gscv, 'mlp')
