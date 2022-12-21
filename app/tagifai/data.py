# data.py
import json
import re
from collections import Counter
from typing import List, Tuple, Dict

import nltk
import numpy as np
import pandas as pd
from config import config
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split


def filter(tag: str, include: List[str] = None) -> str:
    """Determine if a given tag is to be included.

    Args:
        tag (str): Tag to filter against include
        include (List[str], optional): List of tags to include. Defaults to None.

    Returns:
        tag (str): String of tag if included else None.
    """
    if include is None:
        include = []
    if tag not in include:
        tag = None
    return tag


def clean_text(
    text: str,
    lower: bool = True,
    stem: bool = False,
    stopwords: List[str] = config.STOPWORDS,
) -> str:
    """Clean raw text for vectorization and classification.
        Removes stopwords, multiple and trailing spaces and non alphanumeric characters.
        Optionally stems and lowercases text.

    Args:
        text (str): Raw text.
        lower (bool, optional): Whether text should be all lowercase. Defaults to True.
        stem (bool, optional): Whether text should be stemmed. Defaults to False.
        stopwords (List[str], optional): List of stopwords to remove. Defaults to config.STOPWORDS.

    Returns:
        str: Cleaned text.
    """
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
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


def replace_oos_labels(
    df: pd.DataFrame, labels: List[str], label_col: str, oos_label: str = "other"
) -> pd.DataFrame:
    """Replace out of scope (oos) labels in labels col of dataframe.

    Args:
        df (pd.DataFrame): Feature Dataframe with labels column.
        labels (List[str]): List of allowed (non-oos) labels.
        label_col (str): Dataframe column that contains the labels.
        oos_label (str, optional): Replacement label for oos labels. Defaults to "other".

    Returns:
        pd.DataFrame: Feature Dataframe with labels column adjusted to allowed labels.
    """
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: "other" if x in oos_tags else x)
    return df


def replace_minority_labels(
    df: pd.DataFrame, label_col: str, min_freq: int, new_label: str = "other"
) -> pd.DataFrame:
    """Replace minority labels in labels col of dataframe.

    Args:
        df (pd.DataFrame): Feature Dataframe with labels column.
        label_col (str): Dataframe column that contains the labels.
        min_freq (int): Threshold frequency of a given label in data to be included.
            If label count is below, "other" category will be assigned.
        new_label (str, optional): Replacement label for minority labels. Defaults to "other".

    Returns:
        pd.DataFrame: Feature Dataframe with labels column adjusted to allowed labels.
    """
    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(
        label for label in labels.elements() if (labels[label] >= min_freq)
    )
    df[label_col] = df[label_col].apply(
        lambda label: label if label in labels_above_freq else None
    )
    df[label_col] = df[label_col].fillna(new_label)
    return df


def preprocess(
    df: pd.DataFrame, lower: bool, stem: bool, min_freq: int
) -> pd.DataFrame:
    """Preprocess the data by applying text cleaning and oos and minority label replacement.
       Create "text" column with combined text of title and description.

    Args:
        df (pd.DataFrame): Feature Dataframe with label column.
        lower (bool): Whether to lowercase the feature text.
        stem (bool): Whether to stem the feature text.
        min_freq (int): Threshold frequency of a given label in data to be included.
            If label count is below, "other" category will be assigned.

    Returns:
        pd.DataFrame: Feature Dataframe with cleaned "text" column and adjusted labels.
    """
    df["text"] = df.title + " " + df.description  # feature engineering
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other"
    )  # replace OOS labels
    df = replace_minority_labels(
        df=df, label_col="tag", min_freq=min_freq, new_label="other"
    )  # replace labels below min freq

    return df


def get_data_splits(X: pd.Series, y: np.ndarray, train_size: float = 0.7) -> Tuple:
    """Generates balanced datasets for training, validation, and testing.

    Args:
        X (pd.Series): Features Series.
        y (np.ndarray): Array containing the labels.
        train_size (float, optional): Size of the training dataset. Validation and test sizes will be (1-train_size)/2 respectively. Defaults to 0.7.

    Returns:
        Tuple: Tuple of all datasets for train, val and test.
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test


## write own Labelencoder based on scikit-learn
class LabelEncoder:
    """Encode labels into unique indices.
    ```python
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)
    ```
    """

    def __init__(self, class_to_index: Dict = {}) -> None:
        self.class_to_index = class_to_index

    @property
    def classes(self):
        return list(self.class_to_index.keys())

    @property
    def index_to_class(self):
        return {v: k for k, v in self.class_to_index.items()}

    def fit(self, y: List):
        """Fit a list of labels to the encoder.
        Args:
            y (List): raw labels.
        Returns:
            Fitted LabelEncoder instance.
        """
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            self.class_to_index[label] = i
        return self

    def transform(self, y: List) -> np.ndarray:
        """Encode a list of raw labels.
        Args:
            y (List): raw labels.
        Returns:
            np.ndarray: encoded labels as indices.
        """
        _y = [self.class_to_index.get(label) for label in y]
        return np.array(_y)

    def encode(self, y: List):
        """Alternative for transform, implemented as course uses encode/decode"""
        return self.transform(y)

    def inverse_transform(self, y: List) -> List:
        """Decode a list of indices.
        Args:
            y (List): indices.
        Returns:
            List: labels.
        """
        return [self.index_to_class.get(index) for index in y]

    def decode(self, y: List):
        """Alternative for inverse_transform, implemented as course uses encode/decode"""
        return self.inverse_transform(y)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def save(self, fp: str) -> None:
        """Save class instance to JSON file.
        Args:
            fp (str): filepath to save to.
        """
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp: str):
        """Load instance of LabelEncoder from file.
        Args:
            fp (str): JSON filepath to load from.
        Returns:
            LabelEncoder instance.
        """
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def __repr__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"
