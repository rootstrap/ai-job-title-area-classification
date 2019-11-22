import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

from data_process.sentence_normalizer import normalize_sentence
from data_process.data_sets.values_and_labels_dicts import area_value_label_dict, area_label_value_dict
from train_and_test_definition import X_train, y_train, X_test, y_test
from best_params_mlp import best_alpha, best_hidden_layer_sizes, best_random_state, best_use_idf, best_vect_ngram_range

# Count vectorizer
# Transform documents to feature vectors with fit and transform
count_vect = CountVectorizer(ngram_range=best_vect_ngram_range)
X_train_counts = count_vect.fit_transform(X_train)

# td idf transformer to use frecuency of words
tfidf_transformer = TfidfTransformer(use_idf=best_use_idf)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MLPClassifier(
    solver='lbfgs',
    alpha=best_alpha,
    hidden_layer_sizes=best_hidden_layer_sizes,
    random_state=best_random_state
)
clf.fit(X_train_tfidf, y_train)

# Test and show results
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
test_predict = clf.predict(X_test_tfidf)
print(metrics.classification_report(y_test, test_predict, target_names=list(area_label_value_dict.keys())))

# Store the classifier in a .pkl file
with open('mlp.pkl', 'wb') as mlpfile:
    pickle.dump(clf, mlpfile)

# Classify 1000 new examples
with open('example_titles.csv') as f:
    lines = f.readlines()
    lines_without_n = [line.split('\n')[0] for line in lines][1001:2001]
    normalized_s = [normalize_sentence(l1) for l1 in lines_without_n]

    X_counts = count_vect.transform(normalized_s)
    X_tfidf = tfidf_transformer.transform(X_counts)
    y_result = clf.predict(X_tfidf)

    with open('examples_results_mlp.tsv', 'w') as file:
        for i in range(1000):
            file.write(lines_without_n[i])
            file.write("\t")
            file.write(area_value_label_dict[y_result[i]])
            file.write("\n")
        file.close()
