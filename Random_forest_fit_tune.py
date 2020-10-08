import pickle as pck
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.ensemble import HistGradientBoostingClassifier as hgc
from scipy.sparse import csr_matrix
from sklearn.ensemble import GradientBoostingRegressor as gdbc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.decomposition import KernelPCA as pc
from sklearn.decomposition import TruncatedSVD as tsvd
from sklearn.svm import SVC as svc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold as sfk
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression, SGDClassifier
from sklearn.tree import ExtraTreeClassifier
from fit_tune_function import fit_tune_store_sgdcv


est = {'clf__n_estimators': [i*100 for i in range(1, 10)],
       'clf__criterion': ['gini', 'entropy'],
       'clf__oob_score': [True, False],
       'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), ],
       'tfidf__use_idf': (True, False),
       'clf__max_depth': [2, 3, 4, 5],
       'clf__max_features': ['sqrt', 'log2'],
       'clf__class_weight': ['balanced', 'balanced_subsample'],
       'clf__ccp_alpha': [0.1, 0.2, 0.4, 0.5]
       }

clf_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier()),
])

clfs = GridSearchCV(clf_pipeline, est, cv=2, iid=False, n_jobs=-1)
fit_tune_store_sgdcv(clfs, 'Random_forest')


# More Classifier down here

# x_train = pck.load(open('data_process/data_sets/x_train.pkl','rb'))
# y_train = pck.load(open('data_process/data_sets/y_train.pkl','rb'))


# x_train = np.array(x_train)
# y_train = np.array(y_train)

# x_test = pck.load(open('data_process/data_sets/x_test.pkl','rb'))
# y_test = pck.load(open('data_process/data_sets/y_test.pkl','rb'))

# x_test = np.array(x_test)
# y_test = np.array(y_test)


# vectorize = text.TfidfVectorizer()
# vectorizes = text.TfidfTransformer()
# x_train = vectorize.fit_transform(x_train)
# x_test = vectorize.transform(x_test)
# x_train = vectorizes.fit_transform(x_train)
# x_test = vectorizes.transform(x_test)

# # print(x_train)
# # PCA
# # pcd = pc(n_components=len(vectorize.get_feature_names()), kernel='rbf')
# # x_train = pcd.fit_transform(x_train)
# # x_test = pcd.transform(x_test)

# # gvd = tsvd(n_components=10,n_iter=10)
# # x_train = gvd.fit_transform(x_train)
# # x_test = gvd.transform(x_test)

# ExtraTreeClassifier()

# est = {'n_estimators':[i*100 for i in range(1,10)],'criterion':['gini','entropy'],'oob_score':[True,False]}

# forest = RandomForestClassifier(base_estim,n_estimators=500,criterion='gini',n_jobs=-1)
# forest.fit(x_train,y_train)
# target_names = pck.load(open('data_process/data_sets/positions_categories.pkl','rb')).categories.values
# print(classification_report(y_test,forest.predict(x_test),target_names=target_names))

# xgbc = xgboost.XGBClassifier(max_depth=5,n_estimators=400,booster='dart',importance_type='weight')
# xgbc.fit(x_train, y_train)
# print(xgbc.score(x_test, y_test))

# svc = svc(kernel='rbf')
# svc.fit(x_train, y_train)
# print(classification_report(y_test, svc.predict(x_test),target_names=target_names))

# params = {'kernel':('linear','rbf','poly'),'C':[i/1000 for i in range(920, 1000)]}
# ad = ada(base_estimator=forest,n_estimators=200,algorithm='SAMME.R',random_state=42)
# ad.fit(x_train, y_train)
# print(classification_report(y_test,ad.predict(x_test),target_names=target_names))

# # hgbc = hgc(max_iter=200)
# # hgbc.fit(csr_matrix(x_train).todense(),y_train)
# # print(classification_report(y_test,hgbc.predict(csr_matrix(y_test).todense()),target_names=target_names))

# gnbs = gnb()
# gnbs.fit(csr_matrix(x_train).todense(),y_train)
# print(classification_report(y_test,gnbs.predict(csr_matrix(x_test).todense()),target_names=target_names))

# # gpsc = gpc()
# # gpsc.fit(x_train,y_train)
# # print(classification_report(y_test, gpsc.predict(x_test),target_names=target_names))

# sgf = SGDClassifier(loss='log',penalty='elasticnet')
# new_params = {}
# rcc = RidgeClassifierCV(cv=sfk(n_splits=20,shuffle=True,random_state=42),alphas=[1e-3, 1e-2, 1e-1, 1])
# params_E = {}

# sgf.fit(x_train, y_train)
# print(classification_report(y_test,sgf.predict(x_test),target_names=target_names))

# rcc.fit(x_train,y_train)
# print(classification_report(y_test,rcc.predict(x_test),target_names=target_names))

# # gsv = GridSearchCV(estimator=svc,param_grid=params,cv=sfk(n_splits=20,shuffle=True,random_state=42),n_jobs=-1)
# # gsv.fit(x_train, y_train)
# # print(classification_report(y_test,gsv.predict(x_test),target_names=target_names))
# # print(gsv.best_estimator_)

# gsv = GridSearchCV(estimator=ada,param_grid=est,cv=sfk(n_splits=20,shuffle=True,random_state=42),n_jobs=-1)
# gsv.fit(x_train, y_train)
# print(classification_report(y_test,gsv.predict(x_test),target_names=target_names))
# print(gsv.best_estimator_)
