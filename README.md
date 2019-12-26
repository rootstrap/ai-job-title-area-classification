# Classification of job positions by area
This is a project to classify job positions using machine learning, more specifically, supervised learning. The main goal is to get a classifier
that receives a job position in the form of a sentence, written in natural language, for example `CEO and Founder` and returns the job area for that position. The different areas (labels for the classification) are:  
* Business
* Technical
* Sales-Marketing
* Other

In the example, `CEO and Founder` would return `Business`.

There is an analogous project but it classifies according to the level of the position: [Classification of job positions by level](https://github.com/rootstrap/ai-job-title-level-classification).


## Implementation
This project is programed using the [Python language](https://www.python.org). The trained classifiers are implemented in the [Scikit Learn library](https://scikit-learn.org), a set of tools for machine learning in Python. If you use pip and virtual environments, you can install easily the named library: `$pip install -r requirements.txt`.
Two classification classes are studied:  
* [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html?highlight=sgdcl#sklearn.linear_model.SGDClassifier)
* [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=mlpclassifier#sklearn.neural_network.MLPClassifier)

## Process data
Since both classifiers belong to supervised learning, they are trained using manually classified data, that you can see on `data_process/data_sets/classified_titles.tsv`. That is a tab-separated-values file, that has two columns in the form:  
`<job position> | <classification for the job position>`.  
The script `data_process/tsv_to_dataframe.py` takes the `tsv` file and:
1. Generates a [pandas](https://pandas.pydata.org/) dataframe that represents the job positions and corresponding classifications.
2. Split the dataframe into train(X) and test(y) set.
3. Normalizes the dataframe according to a defined criteria.
4. Stores X_train, X_test, y_train, y_test sets.

## Training and tuning
A classifier has:  
* parameters: values that corresponds to the mathematical model, that are adjusted after the training.
* hyper-parameters: values related to the way of training, that are adjusted using a selected part of the training set.  

It's possible to use a `fit` function to train and adjust the params. Besides, scikit learn provides a tool named
 [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), and [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) in order to make exhaustive search to achieve the hyperparams that optimize the results.

## Script execution
The steps are the same as the classification by level:
1. Run `data_process/tsv_file_to_dataframe.py` to extract the data from the tsv file and split the dataset.
2. Run `<clf name>_fit_tune_classifier.py` to fit and tune the classifier. `fit` is to learn and fit the model to the train set, and `tune` is to search for the optimal combination of the hyperparams, the ones that achieves better results(tuning may take a while).
3. Run `<clf name>_test_classifier.py` to test the trained classifiers and show the results. Besides, a classified example set is stored in `test_data/<clf name>_results.tsv`.

Note: `<clf name>` can be `mlp` or `sgd`, depending on the classifier.

## Results
These are the results of each classifier:  

### MLP
```
                coming soon
```

### SGD
```
                   precision    recall  f1-score   support

       Business       0.90      0.89      0.90        63
          Other       0.81      0.89      0.85        44
Sales-Marketing       0.88      0.70      0.78        10
      Technical       0.97      0.94      0.95        33

       accuracy                           0.89       150
      macro avg       0.89      0.85      0.87       150
   weighted avg       0.89      0.89      0.89       150
```
