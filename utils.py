import json
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
stemmer = PorterStemmer()


def clean_text(text, lower=True, stem=False, stopwords=STOPWORDS):
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
            [stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")]
        )

    return text


## write own Labelencoder based on scikit-learn
class LabelEncoder:
    def __init__(self):
        self.class_to_index = {}

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
        _y = []
        for label in y:
            _y.append(self.class_to_index.get(label))
        return np.array(_y)

    def encode(self, y):
        """Alternative for transform, implemented as course uses encode/decode"""
        return self.transform(y)

    def inverse_transform(self, y):
        _y = []
        for index in y:
            _y.append(self.index_to_class.get(index))
        return _y

    def decode(self, y):
        return self.inverse_transform(y)

    def fit_transform(self, y):
        self.fit(y)
        _y = self.transform(y)
        return _y

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
