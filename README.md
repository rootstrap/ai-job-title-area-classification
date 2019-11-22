# Classification of job positions
This is a project to classify job positions using machine learning, more specifically, supervised learning. The main goal is to get a classifier
that receives a job position in the form of a sentence, for example `CEO and Founder` and returns the job area for that position. The different areas (labels for the classification) are:  
* Business
* Technical
* Marketing
* Sales
* Other

In the example, `CEO and Founder` would return `Business`.
Two algorithms are studied:  
* [Stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [Multi-layer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron)


## Implementation
This project is programed using the [Python language](https://www.python.org). The trained classifiers are implemented in the [Scikit Learn library](https://scikit-learn.org), a set of tools for machine learning in Python.

## Process data
Since both algorithms belong to supervised learning, they are trained using manually classified data, that you can see on `data_process/data_sets/classified_titles.tsv`. That is a tab-separated-values file, that has two columns in the form:  
`<job position> | <classification for the job position>`.  
The script `data_process/tsv_file_to_list.py` takes the `tsv` file and creates a list where each element has the form `[position, classification]`.  
The script `data_process/normalize_list.py` takes the named list, and separates it into two new lists: one for the sentences normalized according a defined criteria, and other with the corresponding classification for those sentences. The new lists are stored with the names `normalized_sentences` and `classified_sentences` respectively. To summarize, `normalized_sentences[i]` is the i-th sentence after the normalization, and `classified_sentences[i]` is the corresponding classification.

## Training, testing, tuning
We have now the normalized sentences and the corresponding classification for each sentence. The next thing to do is to split those lists in sets for training (`X_train`, `y_train`) and testing (`X_test`, `y_test`), in order to train the classifier, and to measure the results. This is defined in `train_and_test_definition.py`.  
A classifier has different parameters that are used in its algorithm. It's possible to vary those parameters in order to achieve the bests results in the classification. For each one there is a script named `tuning_<name>_classifier.py` that uses [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), and [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) in order to make exhaustive search to achieve the bests values for the parameters. After that, the combined values that maximize the results, are stored in the `best_params_<name>.py` file.

## General process
Once the best parameters are found, it's possible to get the classifier. The general process, used for both classifiers is:  
* Use `CountVectorizer` that builds a dictionary of features and transforms documents (in this case, sentences) to feature vectors:  
  * `X_train_counts = CountVectorizer(X_train)`.
* Use `TfidfTransformer` that takes into account the frequency of the words inside the sentences:
  * `X_train_tfidf = TfidfTransformer(X_train_counts)`.
* Create the classifier `clf` with the best parameters, and fit it:
  * `clf.fit(X_train_tfidf, y_train)`.

## Summary
So, starting from a `.tsv` with labelled sentences, we build a mlp and sgd classifier. The steps are:
1. Run `data_process/tsv_file_to_list.py` to transform the table into lists.
2. Run `data_process/normalize_list.py` to normalize the list, and transform the labels into integers.
3. Run `tuning_mlp_classifier.py` and `tuning_sgd_classifier.py` to find the best params for each algorithm.
4. Run `mlp_classifier.py` and `sgd_classifier.py` to fit the classifiers, run the tests, and store the results of the classification of new data, to see how it works. Finally, each classifier is dumped using pickle, in a file with its corresponding name. To load the instance, you can execute:  
```
with open('sgd.pkl', 'rb') as sgdfile:
    sgd_loaded = pickle.load(sgdfile)
```

## Results
These are the results of each classifier:  

### MLP
```
                precision    recall  f1-score   support

       Sales       1.00      0.86      0.92         7
   Technical       0.82      0.90      0.86        20
    Business       0.91      0.90      0.90        78
   Marketing       1.00      0.82      0.90        11
       Other       0.78      0.82      0.80        34

    accuracy                           0.87       150
   macro avg       0.90      0.86      0.88       150
weighted avg       0.88      0.87      0.87       150
```

### SGD
```
                precision    recall  f1-score   support

       Sales       1.00      0.86      0.92         7
   Technical       0.88      0.75      0.81        20
    Business       0.83      0.92      0.87        78
   Marketing       1.00      0.82      0.90        11
       Other       0.74      0.68      0.71        34

    accuracy                           0.83       150
   macro avg       0.89      0.80      0.84       150
weighted avg       0.84      0.83      0.83       150
```

The f1 score is a good metric to evaluate the classifier. The closer to value 1, the better. Both classifiers have similar score, but in this case the neural network(mlp classifier) works better. It will adapt better to classify new incoming data.
