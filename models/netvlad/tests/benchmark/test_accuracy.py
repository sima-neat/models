from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def scorer(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def test_listing_splits_database_and_queries_in_official_order(tmp_path: Path, scorer: ModuleType) -> None:
    listing = tmp_path / "pitts30k_test.tsv"
    listing.write_text(
        "database\t000/a.jpg\t10.0\t20.0\ndatabase\t001/b.jpg\t30.0\t40.0\nquery\tqueries_real/000/q.jpg\t11.0\t21.0\n"
    )

    database, database_utm, queries, query_utm = scorer.read_listing(tmp_path)

    assert (database, queries) == (["000/a.jpg", "001/b.jpg"], ["queries_real/000/q.jpg"])
    np.testing.assert_array_equal(database_utm, [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(query_utm, [[11.0, 21.0]])


def test_positives_include_database_images_exactly_25_m_away(scorer: ModuleType) -> None:
    database = np.array([[0.0, 0.0], [25.0, 0.0], [15.0, 20.0], [25.01, 0.0]])

    (matches,) = scorer.positives(database, np.array([[0.0, 0.0]]))

    # 25 m along an axis, and a 15-20-25 triangle, are inside the radius; 25.01 m is not.
    assert matches.tolist() == [0, 1, 2]


def test_nearest_ranks_by_l2_distance_nearest_first(scorer: ModuleType) -> None:
    database = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]], dtype=np.float32)
    queries = np.array([[0.8, 0.6], [0.0, -1.0]], dtype=np.float32)

    ranked = scorer.nearest(database, queries, 3)

    # Squared distances from (0.8, 0.6): 0.08 to (0.6, 0.8), 0.4 to (1, 0), 0.8 to (0, 1).
    assert ranked[0].tolist() == [2, 0, 1]
    # From (0, -1): (1, 0) and (-1, 0) tie at 2, ahead of (0.6, 0.8) at 3.6.
    assert sorted(ranked[1, :2].tolist()) == [0, 3] and ranked[1, 2] == 2


def test_recall_counts_a_query_once_from_its_first_true_match(scorer: ModuleType) -> None:
    predictions = np.tile(np.arange(100), (4, 1))
    truth = [np.array([0]), np.array([3, 70]), np.array([7]), np.array([200])]

    recalls = scorer.recall_at(predictions, truth)

    # Matches first appear at ranks 1, 4 and 8; the fourth query has none in its top 100.
    assert recalls == {1: 0.25, 5: 0.5, 10: 0.75, 20: 0.75, 50: 0.75, 100: 0.75}
