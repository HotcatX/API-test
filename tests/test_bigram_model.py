from collections import Counter

import pytest

from app.bigram_model import BigramModel


def test_successors_are_weighted_by_observed_frequency(monkeypatch):
    model = BigramModel(["we like tea", "we like coffee", "we enjoy class"])

    def choose_successor(options):
        assert Counter(options) == {"like": 2, "enjoy": 1}
        return "enjoy"

    monkeypatch.setattr("app.bigram_model.random.choice", choose_successor)
    assert model.generate_text("we", 2) == "we enjoy"


def test_corpus_entries_do_not_connect_and_terminal_words_are_valid():
    model = BigramModel(["red blue", "green gold", "solo"])
    assert model.generate_text("red", 10) == "red blue"
    assert model.generate_text("blue", 10) == "blue"
    assert model.generate_text("solo", 10) == "solo"


def test_normalization_and_vocabulary():
    model = BigramModel(["WE don’t STOP!"])
    assert model.generate_text("WE!", 10) == "we don't stop"
    assert model.generate_text("DON’T", 2) == "don't stop"
    assert model.vocabulary == ["don't", "stop", "we"]


def test_length_counts_start_word_and_limits_cycles():
    model = BigramModel(["one two one"])
    assert model.generate_text("one", 1) == "one"
    assert model.generate_text("one", 5) == "one two one two one"


@pytest.mark.parametrize("corpus", [[], [""], ["... 123"], "one two", [None]])
def test_invalid_corpus(corpus):
    with pytest.raises(ValueError):
        BigramModel(corpus)


@pytest.mark.parametrize("start", ["", "...", "one two", "missing", None])
def test_invalid_start(start):
    with pytest.raises(ValueError):
        BigramModel(["one two"]).generate_text(start, 3)


@pytest.mark.parametrize("length", [0, -1, 1.5, "3", True])
def test_invalid_length(length):
    with pytest.raises(ValueError):
        BigramModel(["one two"]).generate_text("one", length)
