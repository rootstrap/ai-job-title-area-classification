from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from train_and_test_definition import X_train, y_train

len_n = len(X_train)

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)),
])
text_clf.fit(X_train, y_train)

parameters = {
    'vect__ngram_range': [(1, 1), ],
    'tfidf__use_idf': (True, False),
    'clf__random_state': (1, 21, 33, 42, 88, 160, ),
    'clf__alpha': (1e-2, 1e-3, 1e-4, 0.1, 1e-5, ),
    'clf__hidden_layer_sizes': [(len_n, ), (int(len_n/2), int(len_n/2)), ]
}
gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)

with open('best_params_mlp.py', 'w') as file:
    file.write(f"best_vect_ngram_range = {gs_clf.best_params_['vect__ngram_range']}\n")
    file.write(f"best_use_idf = {gs_clf.best_params_['tfidf__use_idf']}\n")
    file.write(f"best_alpha = {gs_clf.best_params_['clf__alpha']}\n")
    file.write(f"best_random_state = {gs_clf.best_params_['clf__random_state']}\n")
    file.write(f"best_hidden_layer_sizes = {gs_clf.best_params_['clf__hidden_layer_sizes']}\n")
    file.close()
