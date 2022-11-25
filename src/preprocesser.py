
from typing import Union

import pickle

import pandas as pd
import spacy

import tensorflow as tf
from tensorflow import keras


class ReviewPreprocessor:
    """Preprocesser that can be fitted and applied to reviews.

    Arguments
        vocab_size: Max size of tokenizer vocabulary.
        max_seq_length: Max length of padded sentence.
        remove_punc: Whether to remove punctuation.
            While alpha_only also does this, removing punctuation ahead of time
            prevents odd behavior with contractions.
        pad_punc: Pads punctuation on both sides with spaces.
            Only does something if remove_punc=False
        remove_stop: Whether to remove stopwords.
        lemmatize: Whether to lemmatize words.
        alpha_only: Whether to remove non-alphabetical characters
        split_digits: Whether to split numerical values.
            This will truncate to first two digits and use scientific notation
            Examples:
                '1230'   --> '12' 'e2'
                '1'      --> '1' 'e0'
                '155000' --> '15' 'e4'
    """
    def __init__(self, vocab_size, max_seq_length, remove_stop=True,
                 remove_punc=False, pad_punc=True, lemmatize=True,
                 alpha_only=False, split_digits=True):

        # Set parameters
        self.vocab_size = vocab_size
        self.maxlen = max_seq_length

        # Configuration
        self.remove_punc = remove_punc
        self.remove_stop = remove_stop
        self.lemmatize = lemmatize
        self.alpha_only = alpha_only
        self.split_digits = split_digits
        self.pad_punc = pad_punc

        # Tokenizer to be fit
        self.tokenizer = None

        # Load pipeline (requires download)
        #   $ python -m spacy download en_core_web_sm
        self.nlp = spacy.load(
            "en_core_web_sm", disable=["parser", "ner"]
        )

    # Full Pipeline Functions ----------------------------------

    def fit_transform(self, text, batch_size=1_000, n_processes=1):
        text_norm = self.normalize(text, batch_size, n_processes)

        self.tokenizer = keras.preprocessing.text.Tokenizer(self.vocab_size)
        self.tokenizer.fit_on_texts(text_norm)

        return self.to_tensor(text_norm)

    def transform(self, text, batch_size=1_000, n_processes=1):
        if self.tokenizer is None:
            self._raise_token_err()

        text_norm = self.normalize(text, batch_size, n_processes)

        return self.to_tensor(text_norm)

    # Individual Preprocessing Steps -------------------------------

    def normalize(self, text: Union[str, pd.Series], batch_size=1_000, n_processes=1):
        if isinstance(text, str):
            text = pd.Series(text)

        # Replace explicit newlines with spaces
        text = text.str.replace("\\n+", " ", regex=True)

        # Remove punctuation
        if self.remove_punc:
            text = text.str.replace(r"[^\w\s]+", "", regex=True)
        elif self.pad_punc:
            text = text.str.replace(r"([^\w\s])", r" \1 ", regex=True)

        # Convert to lowercase
        text = text.str.lower()

        # Replace multiple spaces with one space
        text = text.str.replace(" {2,}", " ", regex=True)

        # Apply other preprocessing by converting to spacy doc
        results = []
        for doc in self.nlp.pipe(text, batch_size=batch_size, n_process=n_processes):
            results.append(self._process_doc(doc))

        return results

    def fit_tokenizer(self, processed_text):
        self.tokenizer = keras.preprocessing.text.Tokenizer(self.vocab_size)
        self.tokenizer.fit_on_texts(processed_text)

    def to_tensor(self, processed_text):
        if self.tokenizer is None:
            self._raise_token_err()

        result = self.tokenizer.texts_to_sequences(processed_text)

        # Pad sentences to be the same length
        result = keras.preprocessing.sequence.pad_sequences(
            result, maxlen=self.maxlen, padding="post", truncating="post")

        # Convert to tensors for Keras
        return tf.convert_to_tensor(result)

    # I/O ------------------------------------------

    def save_as_pickle(self, filename):
        with open(filename, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_from_pickle(filename):
        with open(filename, "rb") as handle:
            return pickle.load(handle)

    # Helpers -----------------------------------------

    def _process_doc(self, doc):
        """Preprocesses a spacy Doc."""
        output = list()
        for token in doc:
            # Split digit and move to next token if relevant
            if self.split_digits and not self.alpha_only and token.is_digit:
                output.extend(self.split_num(token.text))
                continue

            # Preprocess based on config
            if (not token.is_stop or not self.remove_stop) and \
               (token.is_alpha or not self.alpha_only):
                output.append(token.lemma_ if self.lemmatize else token.text)

        return output

    @staticmethod
    def split_num(num):
        w1, w2 = num[:2], len(num[2:])
        w2 = f"e{w2}"
        return w1, w2

    @staticmethod
    def _raise_token_err():
        err_str = "No tokenizer is defined! Call 'fit_transform' or " \
                  "'fit_tokenizer' on the text with the vocabulary to learn."
        raise TokenizerError(err_str)


class TokenizerError(Exception):
    pass
