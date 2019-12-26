import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_normalizer import normalize_sentence


# Load tsv file into dataframe
data = pd.read_csv('data_sets/classified_titles.tsv', sep='\t')

positions_categories = pd.Categorical(data["Classification"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["Position"],
    positions_categories.codes,
    train_size=0.85,
    stratify=positions_categories.codes
)

X_train = [normalize_sentence(x) for x in X_train]
X_test = [normalize_sentence(x) for x in X_test]

with open('data_sets/x_train.pkl', 'wb') as file:
    pickle.dump(X_train, file)

with open('data_sets/x_test.pkl', 'wb') as file:
    pickle.dump(X_test, file)

with open('data_sets/y_train.pkl', 'wb') as file:
    pickle.dump(y_train, file)

with open('data_sets/y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)

with open('data_sets/positions_categories.pkl', 'wb') as file:
    pickle.dump(positions_categories, file)
