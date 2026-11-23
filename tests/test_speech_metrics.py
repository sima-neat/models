import pytest
from model_report.english_normalizer import EnglishTextNormalizer
from model_report.speech import char_error_rate, normalize, word_error_rate, word_error_rate_as_written


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
    assert word_error_rate(["The cat sat."], ["the cat sat"], "en") == 0


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
    assert word_error_rate(["the cat sat on the mat"], [hypothesis], "en") == pytest.approx(rate)


def test_insertions_can_push_the_rate_above_one():
    assert word_error_rate(["yes"], ["yes yes yes"], "en") == pytest.approx(2.0)


def test_the_rate_is_a_corpus_total_so_long_recordings_weigh_more():
    references = ["a man saw his dog run to the old barn", "hello"]
    hypotheses = ["a man saw his dog run to the old barn", "goodbye"]
    # One error in eleven reference words, not the mean of 0 and 1.
    assert word_error_rate(references, hypotheses, "en") == pytest.approx(1 / 11)


@pytest.mark.parametrize(
    ("text", "scored"),
    [
        ("Twenty-five to thirty", "25 to 30"),
        ("two thousand and eleven", "2011"),
        ("five dollars", "$5"),
        ("twenty five percent", "25%"),
        ("1,000 metres", "1000 meters"),
        ("The travellers weren't there", "the travelers were not there"),
        ("It's Mr. Smith's", "it is mister smith is"),
        ("Café — déjà vu", "cafe deja vu"),
        ("acidic or basic (alkaline) [sic]", "acidic or basic alkaline sic"),
        ('"[w]hile I was not present"', "while i was not present"),
        ("", ""),
    ],
)
def test_english_matches_numbers_contractions_and_spellings_and_keeps_bracketed_words(text, scored):
    assert EnglishTextNormalizer()(text).split() == scored.split()


@pytest.mark.parametrize("language", ["en", "en-US", "EN-gb"])
def test_english_scores_numbers_by_value(language):
    assert word_error_rate(["25 to 30 years"], ["twenty five to thirty years"], language) == 0
    assert char_error_rate(["25 to 30 years"], ["twenty five to thirty years"], language) == 0


def test_other_languages_and_the_as_written_rate_compare_numbers_as_written():
    assert word_error_rate(["fifteen apples"], ["15 apples"], "fr") == pytest.approx(0.5)
    assert word_error_rate_as_written(["fifteen apples"], ["15 apples"]) == pytest.approx(0.5)
    assert word_error_rate_as_written(["travellers"], ["travelers"]) == 1


def test_english_marks_wrong_words_as_errors():
    assert word_error_rate(["fifteen apples"], ["sixteen apples"], "en") == pytest.approx(0.5)


def test_the_language_is_required():
    with pytest.raises(TypeError):
        word_error_rate(["a"], ["a"])
    with pytest.raises(ValueError, match="language"):
        word_error_rate(["a"], ["a"], "")


def test_character_error_rate_ignores_whitespace_for_unspaced_languages():
    assert char_error_rate(["今日は晴れ"], ["今日 は 晴れ"], "ja") == 0
    assert char_error_rate(["今日は晴れ"], ["今日は雨"], "ja") == pytest.approx(2 / 5)


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
        word_error_rate(references, hypotheses, "en")
