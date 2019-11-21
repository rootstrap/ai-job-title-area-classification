from data_process.data_sets.normalized_and_classified_sentences import normalized_sentences, classified_sentences

TEST_SIZE = 150
TRAIN_SIZE = len(normalized_sentences)

X_train = normalized_sentences[0:TRAIN_SIZE-TEST_SIZE]
y_train = classified_sentences[0:TRAIN_SIZE-TEST_SIZE]

X_test = normalized_sentences[TRAIN_SIZE-TEST_SIZE: TRAIN_SIZE + 1]
y_test = classified_sentences[TRAIN_SIZE-TEST_SIZE: TRAIN_SIZE + 1]
