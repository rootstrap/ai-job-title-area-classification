# Classification of job positions by area
This is a project to classify job positions using machine learning, more specifically, supervised learning. The main goal is to get a classifier
that receives a job position in the form of a sentence, written in natural language, for example `CEO and Founder` and returns the job area for that position. The different areas (labels for the classification) are:  
* Business
* Technical
* Marketing
* Sales
* Other


In the example, `CEO and Founder` would return `Business`.
Two classifiers are studied:  
* [Stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [Multi-layer Perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron)

There is an analogous project but it classifies according to the level of the position: [Classification of job positions by level](https://github.com/rootstrap/ai-job-title-level-classification).


## Implementation
This project is programed using the [Python language](https://www.python.org). The trained classifiers are implemented in the [Scikit Learn library](https://scikit-learn.org), a set of tools for machine learning in Python. If you use pip and virtual environments, you can install easily the named library: `$pip install -r requirements.txt`.

## Process data
Since both classifiers belong to supervised learning, they are trained using manually classified data, that you can see on `data_process/data_sets/classified_titles.tsv`. That is a tab-separated-values file, that has two columns in the form:  
`<job position> | <classification for the job position>`.  
The script `data_process/tsv_file_to_list.py` takes the `tsv` file and creates a list where each element has the form `[position, classification]`.  
The script `data_process/normalize_list.py` takes the named list, and separates it into two new lists: one for the sentences normalized according a defined criteria, and other with the corresponding classification for those sentences. The new lists are stored with the names `normalized_sentences` and `classified_sentences` respectively. To summarize, `normalized_sentences[i]` is the i-th sentence after the normalization, and `classified_sentences[i]` is the corresponding classification.

## Training, tuning, testing
We have now the normalized sentences and the corresponding classification for each sentence. The next thing to do is to split those lists in sets for training (`X_train`, `y_train`) and testing (`X_test`, `y_test`), in order to fit and tune the classifier, and to measure the results. This is defined in `train_and_test_definition.py`.  
A classifier has:  
* parameters: values that corresponds to the mathematical model, that are adjusted after the training.
* hyper-parameters: values related to the way of training, that are adjusted using a selected part of the training set.  

It's possible to use a `fit` function to train and adjust the params. Besides, scikit learn provides a tool named
 [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), and [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) in order to make exhaustive search to achieve the hyperparams that optimizes the results.

## General process
The general process, used for both classifiers is:  
* Use `CountVectorizer` that builds a dictionary of features and transforms documents (in this case, sentences) to feature vectors:  
  * `X_train_counts = CountVectorizer(X_train)`.
* Use `TfidfTransformer` that takes into account the frequency of the words inside the sentences:
  * `X_train_tfidf = TfidfTransformer(X_train_counts)`.
* Create the classifier `clf` with the best parameters, and fit it:
  * `clf.fit(X_train_tfidf, y_train)`.

## Script execution
The steps are the same as the classification by area:
1. Run `data_process/tsv_file_to_list.py` to transform the table into lists.
2. Run `data_process/normalize_list.py` to normalize the list, and transform the labels into integers.
3. Run `<clf name>_fit_tune_classifier.py` to fit and tune the classifier. `fit` is to learn and fit the model to the train set, and `tune` is to search for the optimal combination of the hyperparams, the ones that achieves better results(tuning may take a while).
4. Run `<clf name>_test_classifier.py` to test the trained classifiers and show the results. Besides, a classified example set is stored in `test_data/<clf name>_results.tsv`.

Note: `<clf name>` can be `mlp` or `sgd`, depending on the classifier.

## Results
These are the results of each classifier:  

### MLP
```
                precision    recall  f1-score   support

       Sales       1.00      1.00      1.00         5
   Technical       0.94      0.73      0.82        22
    Business       0.79      0.94      0.86        65
   Marketing       1.00      0.78      0.88         9
       Other       0.86      0.78      0.82        49

    accuracy                           0.85       150
   macro avg       0.92      0.84      0.87       150
weighted avg       0.86      0.85      0.85       150
```

### SGD
```
                precision    recall  f1-score   support

       Sales       1.00      1.00      1.00         5
   Technical       0.88      0.68      0.77        22
    Business       0.77      0.95      0.85        65
   Marketing       0.88      0.78      0.82         9
       Other       0.90      0.71      0.80        49

    accuracy                           0.83       150
   macro avg       0.88      0.83      0.85       150
weighted avg       0.84      0.83      0.82       150
```

The f1 score is a good metric to evaluate the classifier. The closer to value 1, the better. Both classifiers have similar score, but in this case the neural network (mlp classifier) works better. It will adapt better to classify new incoming data.
