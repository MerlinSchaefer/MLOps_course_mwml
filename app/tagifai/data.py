# data.py
import json
import re
from collections import Counter
import nltk
import numpy as np
from config import config
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split


def filter(tag, include=None):
    """Determine if a given tag is to be included."""
    if include is None:
        include = []
    if tag not in include:
        tag = None
    return tag


def clean_text(text, lower=True, stem=False, stopwords=config.STOPWORDS):
    """Apply basic cleaning functions to raw text"""
    if lower:
        text = text.lower()

    # remove stopwords
    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)
    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        text = " ".join(
            [config.stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")]
        )

    return text


def replace_oos_labels(df, labels, label_col, oos_label="other"):
    """Replace out of scope (oos) labels"""
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: "other" if x in oos_tags else x)
    return df


def replace_minority_labels(df, label_col, min_freq, new_label="other"):
    """Replace minority labels with other label"""
    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(
        label for label in labels.elements() if (labels[label] >= min_freq)
    )
    df[label_col] = df[label_col].apply(
        lambda label: label if label in labels_above_freq else None
    )
    df[label_col] = df[label_col].fillna(new_label)
    return df


def preprocess(df, lower, stem, min_freq, accepted_tags=config.ACCEPTED_TAGS):
    """Preprocess the data."""
    df["text"] = f"{df.title} {df.description}"
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text

    # Replace OOS tags with `other`
    df = replace_oos_labels(
        df, labels=accepted_tags, label_col="tag", oos_label="other"
    )

    # Replace tags below min_freq with `other`
    df = replace_minority_labels(
        df, label_col="tag", min_freq=min_freq, new_label="other"
    )

    return df


def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test


## write own Labelencoder based on scikit-learn
class LabelEncoder:
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index

    @property
    def classes(self):
        return list(self.class_to_index.keys())

    @property
    def index_to_class(self):
        return {v: k for k, v in self.class_to_index.items()}

    def fit(self, y):
        """Fit the label encoder"""
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            self.class_to_index[label] = i
        return self

    def transform(self, y):
        """Transform labels to normalized encoding"""
        _y = [self.class_to_index.get(label) for label in y]
        return np.array(_y)

    def encode(self, y):
        """Alternative for transform, implemented as course uses encode/decode"""
        return self.transform(y)

    def inverse_transform(self, y):
        return [self.index_to_class.get(index) for index in y]

    def decode(self, y):
        return self.inverse_transform(y)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def __repr__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"
