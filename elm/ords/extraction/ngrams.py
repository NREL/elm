# -*- coding: utf-8 -*-
"""ELM Ordinance ngram text validation

This check helps validate that the LLM extracted text from the original
document and did not make it up itself.
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams


nltk.download("punkt")
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))
PUNCTUATIONS = {'"', ".", "(", ")", ",", "?", ";", ":", "''", "``"}


def _check_word(word):
    """``True`` if a word is not a stop word or a punctuation."""
    return word not in STOP_WORDS and word not in PUNCTUATIONS


def _filtered_words(sentence):
    """Filter out common words and punctuations."""
    return [
        word.casefold()
        for word in word_tokenize(sentence)
        if _check_word(word.casefold())
    ]


def convert_text_to_sentence_ngrams(text, n):
    """Convert input text to a list of ngrams.

    The text is first split byu sentence, after which each sentence is
    converted into ngrams. The ngrams for all sentences are combined and
    returned.

    Parameters
    ----------
    text : str
        Input text containing one or more sentences.
    n : int
        Number of words to include per ngram.

    Returns
    -------
    list
        List of tuples, where each tuple is an ngram from the original
        text.
    """
    all_ngrams = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = _filtered_words(sentence)
        all_ngrams += list(ngrams(words, n))
    return all_ngrams


def sentence_ngram_containment(original, test, n):
    """Fraction of sentence ngrams from the test text found in the original.

    Parameters
    ----------
    original : str
        Original (superset) text. Ngrams from the `test` text will be
        checked against this text.
    test : str
        Test (sub) text. Ngrams from this text will be searched for in
        the original text, and the fraction of these ngrams that are
        found in the original text will be returned.
    n : int
        Number of words to include per ngram.

    Returns
    -------
    float
        Fraction of ngrams from the `test` input that were found in the
        `original` text. Always returns ``True`` if test has no ngrams.
    """
    ngrams_test = convert_text_to_sentence_ngrams(test, n)
    num_test_ngrams = len(ngrams_test)
    if not num_test_ngrams:
        return True

    ngrams_original = set(convert_text_to_sentence_ngrams(original, n))
    num_ngrams_found = sum(t in ngrams_original for t in ngrams_test)
    return num_ngrams_found / num_test_ngrams
