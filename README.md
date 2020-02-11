# Path Explain

A repository for explaining feature importances and feature interactions in deep neural networks using path attribution methods.

This repository contains tools to interpret and explain machine learning models using [Integrated Gradients](https://arxiv.org/abs/1703.01365) and [Expected Gradients](https://arxiv.org/abs/1906.10670). In addition, it contains code to explain _interactions_ in deep networks using Integrated Hessians and Expected Hessians - methods that we introduced in our most recent paper: "Explaining Explanations: Axiomatic Feature Interactions for Deep Networks" (arXiv link will be up shortly). 

This repository contains two important directories: the `path_explain` directory, which contains the packages used to interpret and explain machine learning models, and the `examples` directory, which contains many examples using the `path_explain` module to explain different models on different data types. 

## Installation

The easiest way to install this package is by using pip:
```
pip install path-explain
```
Alternatively, you can clone this repository to re-run and explore the examples provided.

## Comptability
This package was written to support TensorFlow 2.0 (in eager execution mode) with Python 3. We have no current plans to support earlier versions of TensorFlow or Python. 

## Examples

For a simple, quick example to get started using this repository, see the `example_usage.ipynb` notebook in the top-level directory of this repository. It gives an overview of the functionality provided by this repository. For more advanced examples, keep reading on. 

### Tabular Data using Expected Gradients

Our repository can easily be adapted to explain attributions and interactions learned on tabular data. 
```python
# other import statements...
from path_explain import PathExplainerTF, scatter_plot, summary_plot

### Code to train a model would go here
x_train, y_train, x_test, y_test = datset()
model = ...
model.fit(x_train, y_train, ...)
###

### Generating attributions using expected gradients
attributions = explainer.attributions(inputs=x_test,
                                      baseline=x_train,
                                      batch_size=100,
                                      num_samples=200,
                                      use_expectation=True,
                                      output_indices=0)
###

### Generating interactions using expected hessians
interactions = explainer.interactions(inputs=x_test,
                                      baseline=x_train,
                                      batch_size=100,
                                      num_samples=200,
                                      use_expectation=True,
                                      output_indices=0)
###
```

Once we've generated attributions and interactions, we can use the provided plotting modules to help visualize them:
```python
### First we need a list of strings denoting the name of each feature
feature_names = ...
###

summary_plot(attributions=attributions,
             feature_values=x_test,
             feature_names=feature_names,
             plot_top_k=10)
```

### Language
