# Image Captioning with CNN and RNN

Tensorflow/Keras implementation of an image captioning neural network, using CNN and RNN.

## Description

This is an unofficial implementation of the image captioning model proposed in the paper ["Show and tell: A neural image caption generator."](https://arxiv.org/abs/1411.4555). 

This implementation is a faithful reproduction of the technique proposed in the paper, where during training, we provide the model with **examples composed of an image, its caption (inputs) and the caption with words shifted one position to the left (ground-truth)**; this approach is faster during training and easier to understand than the other possible approach based on feeding prefixes of the captions to the model and predicting a single word for prefix example. The input caption is needed since we apply **teacher forcing** while training.

The following diagram shows the components of the model during training (in red the losses for each timestep).

![diagram of the model during training](reports/figures/training_model.png)

## Getting Started

Install the Python requirements in your virtual environment.

The dataset used is [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k), containing 8000 images, where each image is associated with five different captions.

Enter the `src` dir and run:
```
python -m data.download_dataset
```
to download the dataset from Kaggle. You will need to setup your Kaggle username and API key locally ([instructions](https://www.kaggle.com/datasets/adityajn105/flickr8k)). Check if the dataset has been correctly downloaded inside `/data/raw`.

You can then run:
```
python -m data.make_dataset
```
to transform the captions into sequences using the Spacy custom tokenizer contained in this project and store the train/val/test sets inside `/data/processed`. The splits will be stored as TFRecords containing examples made of the images together with their five captions. Using the TFRecords will make it easy to use Tensorflow Data API to train and evaluate the model. Each TFRecords contains 200 examples. The ids of the images in the split are stored in the `.txt` files in `data/raw`, already loaded in the repository. This should be the official split proposed by the creators of Flickr8k. 

You can modify the model in the source at `models/model.py` and change the training parameters in `models/train_model.py`. Then training can be started with:
```
python -m models.train_model
```
and it will show the loss and accuracy (considering teacher forcing) of the model at training time. It will also compute the same values on the validation dataset. The weights of the trained model are saved inside `/models/weights`.

To make predictions over custom images, enter your `*.jpg` files inside the `/data/custom` directory. Then run:
```
python -m models.predict_model
```
to show the generated captions in terminal.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## References

[1] Vinyals, Oriol, et al. ["Show and tell: A neural image caption generator."](https://arxiv.org/abs/1411.4555) Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[2] Tanti, Marc, Albert Gatt, and Kenneth P. Camilleri. ["Where to put the image in an image caption generator."](https://arxiv.org/abs/1703.09137) Natural Language Engineering 24.3 (2018): 467-489.
