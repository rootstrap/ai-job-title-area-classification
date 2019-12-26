import pickle
from sklearn import metrics

from data_process.sentence_normalizer import normalize_sentence


def load_instances(clf_type):
    clf = pickle.load(open(f'{clf_type}_instances/{clf_type}_clf.pkl', 'rb'))
    count_vect = pickle.load(open(f'{clf_type}_instances/{clf_type}_count_v.pkl', 'rb'))
    tfidf_transformer = pickle.load(open(f'{clf_type}_instances/{clf_type}_tfidf_t.pkl', 'rb'))
    return clf, count_vect, tfidf_transformer


def test_classifier(clf, test_predict):
    y_test = pickle.load(open('data_process/data_sets/y_test.pkl', 'rb'))
    positions_categories = pickle.load(open('data_process/data_sets/positions_categories.pkl', 'rb'))
    print(metrics.classification_report(y_test, test_predict, target_names=positions_categories.categories))


def test_with_examples(clf_type, clf, count_vect, tfidf_transformer):
    positions_categories = pickle.load(open('data_process/data_sets/positions_categories.pkl', 'rb'))
    # Classify new examples
    with open('test_data/example_titles.csv') as f:
        lines = f.readlines()
        lines_without_n = [line.split('\n')[0] for line in lines][4001:5001]
        normalized_s = [normalize_sentence(l1) for l1 in lines_without_n]

        X_counts = count_vect.transform(normalized_s)
        X_tfidf = tfidf_transformer.transform(X_counts)
        y_result = clf.predict(X_tfidf)

    with open(f'test_data/{clf_type}_results.tsv', 'w') as file:
        for job_pos, classified_job in zip(lines_without_n, y_result):
            file.write(job_pos)
            file.write("\t")
            file.write(positions_categories.categories[classified_job])
            file.write("\n")
        file.close()
