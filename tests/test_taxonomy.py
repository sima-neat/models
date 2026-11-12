from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from model_registry.errors import ValidationError
from model_registry.taxonomy import canonical_taxonomy_document, load_taxonomy, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "taxonomies/model-taxonomy.yaml"


@pytest.mark.parametrize("description", [None, "", "  ", 42, ["description"]])
def test_family_requires_a_nonempty_text_description(tmp_path: Path, description: object) -> None:
    taxonomy = yaml.safe_load(TAXONOMY.read_text())
    entry = taxonomy["architectures"]["resnet"]
    if description is None:
        del entry["description"]
    else:
        entry["description"] = description
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(taxonomy))
    with pytest.raises(ValidationError, match=r"architectures.resnet.description must be non-empty"):
        load_taxonomy(path)


def test_family_description_is_catalog_metadata(tmp_path: Path) -> None:
    taxonomy = load_taxonomy(TAXONOMY)
    document = json.loads(json.dumps(canonical_taxonomy_document(taxonomy)))
    assert document == taxonomy
    assert all(entry["description"].strip() for entry in document["architectures"].values())

    document["architectures"]["resnet"]["description"] = "A new family description."
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(document))
    assert taxonomy_sha256(load_taxonomy(path)) != taxonomy_sha256(taxonomy)


def test_description_is_only_allowed_on_families(tmp_path: Path) -> None:
    taxonomy = load_taxonomy(TAXONOMY)
    taxonomy["tasks"]["image_classification"]["description"] = "Not a family."
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(taxonomy))
    with pytest.raises(ValidationError, match="unknown fields: description"):
        load_taxonomy(path)
