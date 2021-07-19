# Path Explain

A repository for explaining feature importances and feature interactions in deep neural networks using path attribution methods.

This repository contains tools to interpret and explain machine learning models using [Integrated Gradients](https://arxiv.org/abs/1703.01365) and [Expected Gradients](https://arxiv.org/abs/1906.10670). In addition, it contains code to explain _interactions_ in deep networks using Integrated Hessians and Expected Hessians - methods that we introduced in our most recent paper: ["Explaining Explanations: Axiomatic Feature Interactions for Deep Networks"](https://www.jmlr.org/papers/v22/20-1223.html). If you use our work to explain your networks, please cite this paper.

```
@article{janizek2020explaining,
  author  = {Joseph D. Janizek and Pascal Sturmfels and Su-In Lee},
  title   = {Explaining Explanations: Axiomatic Feature Interactions for Deep Networks},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {104},
  pages   = {1-54},
  url     = {http://jmlr.org/papers/v22/20-1223.html}
}
```

This repository contains two important directories: the `path_explain` directory, which contains the packages used to interpret and explain machine learning models, and the `examples` directory, which contains many examples using the `path_explain` module to explain different models on different data types.

## Installation

The easiest way to install this package is by using pip:
```
pip install path-explain
```
Alternatively, you can clone this repository to re-run and explore the examples provided.

## Compatibility
This package was written to support TensorFlow 2.0 (in eager execution mode) with Python 3. We have no current plans to support earlier versions of TensorFlow or Python.

## API
Although we don't yet have formal API documentation, the underlying code does a pretty good job at explaining the API. See the code for generating [attributions](https://github.com/suinleelab/path_explain/blob/master/path_explain/explainers/path_explainer_tf.py#L302) and [interactions](https://github.com/suinleelab/path_explain/blob/master/path_explain/explainers/path_explainer_tf.py#L445) to better understand what the arguments to these functions mean.

## Examples

For a simple, quick example to get started using this repository, see the `example_usage.ipynb` notebook in the top-level directory of this repository. It gives an overview of the functionality provided by this repository. For more advanced examples, keep reading on.

### Tabular Data using Expected Gradients and Expected Hessians

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
explainer = PathExplainerTF(model)
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

Once we've generated attributions and interactions, we can use the provided plotting modules to help visualize them. First we plot a summary of the top features and their attribution values:
```python
### First we need a list of strings denoting the name of each feature
feature_names = ...
###

summary_plot(attributions=attributions,
             feature_values=x_test,
             feature_names=feature_names,
             plot_top_k=10)
```
![Heart Disease Summary Plot](/images/heart_disease.png)

Second, we plot an interaction our model has learned between maximum achieved heart rate and gender:
```python
scatter_plot(attributions=attributions,
             feature_values=x_test,
             feature_index='max. achieved heart rate',
             interactions=interactions,
             color_by='is male',
             feature_names=feature_names,
             scale_y_ind=True)
```
![Interaction: Heart Rate and Gender](/images/max_heart_rate.png)

The model used to generate the above interactions is a two layer neural network trained on the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). Interactions learned by this model were featured in our paper. To learn more about this particular model and the experimental setup, see [the notebook used to train and explain the model](https://github.com/suinleelab/path_explain/blob/master/examples/tabular/heart_disease/attributions.ipynb).


### Explaining an NLP model using Integrated Gradients and Integrated Hessians
As discussed in our paper, we can use Integrated Hessians to get interactions in language models. We explain a transformer from the [HuggingFace Transformers Repository](https://github.com/huggingface/transformers).
```python
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, \
                         DistilBertConfig, glue_convert_examples_to_features, \
                         glue_processors

# This is a custom explainer to explain huggingface models
from path_explain import EmbeddingExplainerTF, text_plot, matrix_interaction_plot, bar_interaction_plot

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)

### Some custom code to fine-tune the model on a sentiment analysis task...
max_length = 128
data, info = tensorflow_datasets.load('glue/sst-2', with_info=True)
train_dataset = glue_convert_examples_to_features(data['train'],
                                                  tokenizer,
                                                  max_length,
                                                  'sst-2)
valid_dataset = glue_convert_examples_to_features(data['validation'],
                                                  tokenizer,
                                                  max_length,
                                                  'sst-2')
...
### we won't include the whole fine-tuning code. See the HuggingFace repository for more.

### Here we define functions that represent two pieces of the model:
### embedding and prediction
def embedding_model(batch_ids):
    batch_embedding = model.distilbert.embeddings(batch_ids)
    return batch_embedding

def prediction_model(batch_embedding):
    # Note: this isn't exactly the right way to use the attention mask.
    # It should actually indicate which words are real words. This
    # makes the coding easier however, and the output is fairly similar,
    # so it suffices for this tutorial.
    attention_mask = tf.ones(batch_embedding.shape[:2])
    attention_mask = tf.cast(attention_mask, dtype=tf.float32)
    head_mask = [None] * model.distilbert.num_hidden_layers

    transformer_output = model.distilbert.transformer([batch_embedding, attention_mask, head_mask], training=False)[0]
    pooled_output = transformer_output[:, 0]
    pooled_output = model.pre_classifier(pooled_output)
    logits = model.classifier(pooled_output)
    return logits
###

### We need some data to explain
for batch in valid_dataset.take(1):
    batch_input = batch[0]

batch_ids = batch_input['input_ids']
batch_embedding = embedding_model(batch_ids)

baseline_ids = np.zeros((1, 128), dtype=np.int64)
baseline_embedding = embedding_model(baseline_ids)
###

### We are finally ready to explain our model
explainer = EmbeddingExplainerTF(prediction_model)
attributions = explainer.attributions(inputs=batch_embedding,
                                      baseline=baseline_embedding,
                                      batch_size=32,
                                      num_samples=256,
                                      use_expectation=False,
                                      output_indices=1)
###

### For interactions, the hessian is rather large so we use a very small batch size
interactions = explainer.interactions(inputs=batch_embedding,
                                      baseline=baseline_embedding,
                                      batch_size=1,
                                      num_samples=256,
                                      use_expectation=False,
                                      output_indices=1)
###
```
We can plot the learned attributions and interactions as follows. First we plot the attributions:
```python
### First we need to decode the tokens from the batch ids.
batch_sentences = ...
### Doing so will depend on how you tokenized your model!

text_plot(batch_sentences[0],
          attributions[0],
          include_legend=True)
```
![Showing feature attributions in text](/images/little_to_love_text.png)

Then we plot the interactions:
```python
bar_interaction_plot(interactions[0],
                     batch_sentences[0],
                     top_k=5)
```
![Showing feature interactions in text](/images/little_to_love_bar.png)

If you would rather plot the full matrix of attributions rather than the top interactions in a bar plot, our package also supports this. First we show the attributions:
```python
text_plot(batch_sentences[1],
          attributions[1],
          include_legend=True)
```
![Showing additional attributions](/images/painfully_funny_text.png)

And then we show the full interaction matrix. Here we've zeroed out the diagonals so you can better see the off-diagonal terms.
```python
matrix_interaction_plot(interaction_list[1],
                        token_list[1])
```
![Showing the full matrix of feature interactions](/images/painfully_funny_matrix.png)

This example - interpreting [DistilBERT](https://arxiv.org/abs/1910.01108) - was also featured in our paper. You can examine the setup more [here](https://github.com/suinleelab/path_explain/tree/master/examples/natural_language/transformers). For more examples, see the `examples` directory in this repository.
