import pytest
from model_report.speech import char_error_rate, normalize, word_error_rate


@pytest.mark.parametrize(
    ("text", "scored"),
    [
        ("Hello, World!", "hello world"),
        ("  The\tcat\n sat.  ", "the cat sat"),
        ("Don't stop", "don t stop"),
        ("“Quoted” — and … more", "quoted and more"),
        ("Ｆｕｌｌ-width ＡＢＣ", "full width abc"),
        ("15 + 3 = 18", "15 3 18"),
        ("नमस्ते दुनिया।", "नमस्ते दुनिया"),
    ],
)
def test_normalize_drops_case_punctuation_and_symbols_but_keeps_letters_marks_and_digits(text, scored):
    assert normalize(text) == scored


def test_identical_transcripts_score_zero_after_normalization():
    assert word_error_rate(["The cat sat."], ["the cat sat"]) == 0


@pytest.mark.parametrize(
    ("hypothesis", "rate"),
    [
        ("the dog sat on the mat", 1 / 6),
        ("the cat on the mat", 1 / 6),
        ("the cat sat on the big mat", 1 / 6),
        ("", 1.0),
        ("cat the sat on the mat", 2 / 6),
    ],
)
def test_word_error_rate_counts_substitutions_deletions_and_insertions(hypothesis, rate):
    assert word_error_rate(["the cat sat on the mat"], [hypothesis]) == pytest.approx(rate)


def test_insertions_can_push_the_rate_above_one():
    assert word_error_rate(["yes"], ["yes yes yes"]) == pytest.approx(2.0)


def test_the_rate_is_a_corpus_total_so_long_recordings_weigh_more():
    references = ["one two three four five six seven eight nine ten", "hello"]
    hypotheses = ["one two three four five six seven eight nine ten", "goodbye"]
    # One error in eleven reference words, not the mean of 0 and 1.
    assert word_error_rate(references, hypotheses) == pytest.approx(1 / 11)


def test_numbers_are_compared_as_written():
    assert word_error_rate(["fifteen apples"], ["15 apples"]) == pytest.approx(0.5)


def test_character_error_rate_ignores_whitespace_for_unspaced_languages():
    assert char_error_rate(["今日は晴れ"], ["今日 は 晴れ"]) == 0
    assert char_error_rate(["今日は晴れ"], ["今日は雨"]) == pytest.approx(2 / 5)


@pytest.mark.parametrize(
    ("references", "hypotheses", "message"),
    [
        (["a", "b"], ["a"], "2 references but 1 hypotheses"),
        (["!?"], ["anything"], "no words"),
        ([], [], "no words"),
    ],
)
def test_unscorable_inputs_are_rejected(references, hypotheses, message):
    with pytest.raises(ValueError, match=message):
        word_error_rate(references, hypotheses)
