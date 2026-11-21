from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def test_local_markdown_links_resolve() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    failures = []
    for document in markdown_files:
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            path = target.split("#", 1)[0]
            if not path or "://" in path or path.startswith("mailto:"):
                continue
            resolved = (document.parent / path).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not failures, "broken local Markdown links:\n" + "\n".join(failures)


def test_developer_index_covers_the_contribution_path() -> None:
    text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for guide in ("adding-a-model.md", "metadata-schema.md", "local-development.md", "contributing.md"):
        assert f"]({guide})" in text


def test_developer_sidebar_includes_every_article() -> None:
    sidebar = (ROOT / "website" / "sidebars.ts").read_text(encoding="utf-8")

    for article in sorted((ROOT / "docs").glob("*.md")):
        doc_id = article.stem
        assert f"'{doc_id}'" in sidebar, f"docs/{article.name} is missing from the developer sidebar"


def test_contributor_docs_distinguish_pr_checks_from_trusted_automation() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "vulcan-model-orchestrator.yml").read_text(encoding="utf-8")
    contributing_words = " ".join(contributing.split())

    assert "Pull requests do **not** run recipe or surgery code" in overview
    assert "## After merge: trusted staging automation" in contributing
    assert "A successful compile does not by itself mean the benchmark succeeded" in contributing_words
    assert "pull_request" not in workflow


def test_local_qualification_is_required_before_submission() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    overview_words = " ".join(overview.split())

    assert "submission prerequisites, not optional CI substitutes" in overview_words
    assert "complete the entire loop before submitting a model pull request" in local
    assert "Do not start submission until every supported component/profile pair" in contributing
    assert "checks not run locally" not in contributing


def test_contribution_guides_have_single_workflow_owners() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    authoring = (ROOT / "docs" / "adding-a-model.md").read_text(encoding="utf-8")
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    overview_words = " ".join(overview.split())

    assert "owns the tool installation and complete test sequence" in overview_words
    assert "sima-cli neat install --stg models-utils/model-registry" not in overview
    assert "model-registry matrix --repository" not in authoring
    assert "model-registry verify-mpk" not in authoring
    assert "does not repeat the local qualification procedure" in contributing
    assert "model-registry validate models/<model-id>" not in contributing
    assert "## Local-first qualification loop" in local


def test_core_page_titles_match_the_journey_navigation() -> None:
    sidebar = (ROOT / "website" / "sidebars.ts").read_text(encoding="utf-8")
    component = (ROOT / "website" / "src" / "components" / "ContributionPath.tsx").read_text(encoding="utf-8")
    expected = {
        "adding-a-model.md": ("1", "Define the model contract"),
        "local-development.md": ("2", "Qualify locally"),
        "contributing.md": ("3", "Submit for review"),
    }

    for filename, (step, title) in expected.items():
        article = (ROOT / "docs" / filename).read_text(encoding="utf-8")
        assert article.startswith(f"# {title}\n")
        assert f"label: '{step}. {title}'" in sidebar
        assert f"guideLabel: '{title}'" in component


def test_dataset_guide_covers_artifactory_access_and_registration() -> None:
    text = (ROOT / "docs" / "datasets.md").read_text(encoding="utf-8")

    for expected in (
        "## Set up Artifactory access first",
        "dataset-registry resolve imagenet-2012",
        'jq -e \'.provider == "artifactory"\'',
        "## Introduce a dataset or variant that does not exist",
        "catalog/datasets.yaml",
        "artifactory_sources",
        "Dataset pull-request CI is intentionally credential-free",
        "## Troubleshooting access",
    ):
        assert expected in text

    assert "Do not add a one-off download path to the model recipe" in text


def test_contribution_path_wraps_without_horizontal_scrolling() -> None:
    component = (ROOT / "website" / "src" / "components" / "ContributionPath.tsx").read_text(encoding="utf-8")
    css = (ROOT / "website" / "src" / "css" / "custom.css").read_text(encoding="utf-8")

    assert "<ol" in component
    assert "<svg" not in component
    assert "'six five four'" in css
    assert "@container (max-width: 700px)" in css
    assert "@container (max-width: 480px)" in css


def test_local_guide_passes_scoped_calibration_to_the_recipe() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    assert "fetch-calibration \\" in text
    assert "models/<model-id> --build-profile <profile-id> --component <component-id> --json" in text
    assert 'recipe_args=(--dataset "$calibration_path")' in text
    assert '"${recipe_args[@]}"' in text
    assert "## Measure declared accuracy locally" in text


def test_local_guide_uses_supported_validation_commands() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    assert "sima-cli neat install --stg models-utils/model-registry@develop:latest" in text
    assert "python -m pip install -e" not in text
    for command in (
        "model-registry validate ",
        "model-registry validate-models ",
        "model-registry fetch ",
        "model-registry fetch-calibration ",
        "model-registry matrix ",
        "model-registry resolve-compiler ",
        "model-registry verify-mpk ",
    ):
        assert command in text


def test_local_guide_covers_the_complete_local_qualification_loop() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    for expected in (
        "Compile locally from `model.yaml`",
        "Model Compiler Python APIs available",
        "compile.py --model <verified-entrypoint>",
        "model-registry verify-mpk",
        "Benchmark locally on a Modalix DevKit",
        "python models/resnet_50/scripts/performance.py",
        "model-report.json",
        "Finish the local review",
    ):
        assert expected in text

    assert "Compilation is intentionally model-owned" in text
    assert "does not provide a public compile command" in text.replace("\n", " ")
    assert "Keep the report and log outside Git" in text
    assert "Vulcan" not in text
    assert "What CI repeats" not in text
    assert "GitHub-hosted runner" not in text


def test_docs_keep_author_tools_separate_from_ci_publication() -> None:
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    benchmarking = (ROOT / "docs" / "benchmarking.md").read_text(encoding="utf-8")

    assert "models-utils/model-registry@develop:latest" in local
    assert "Compilation is intentionally model-owned" in local
    assert "Model authors do not publish benchmark results directly" in benchmarking
    assert "reserved for CI" in contributing
    assert "git push --set-upstream origin HEAD" in contributing
    assert "authors do not invoke those publication steps themselves" in contributing

    author_guides = local + benchmarking
    for command in (
        "model-registry publish ",
        "model-registry publish-benchmark ",
        "model-registry publish-catalog ",
        "model-registry package ",
    ):
        assert command not in author_guides


# The structure of the documentation, one reader per tree: docs/ is the public site for people, agents/ serves
# agents working in this repository, and skills/ serves customers' agents and ships alone. AGENTS.md indexes all
# three. agents/rules.md states the rules for skills/, and agents/references/writing-docs.md the rules for docs/.

MODELS_REPOSITORY_LINK = re.compile(r"https://github\.com/sima-neat/models/(?:blob|tree|edit)/[^/]+/([^)\s]+)")
INTERNAL_TREES = ("AGENTS.md", "agents/", "design-doc/")
# AGENTS.md is loaded into every agent session, so its length is a cost on every task.
AGENTS_LINE_BUDGET = 130
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
# A reference to the files written for agents in this repository: agents/ or AGENTS.md at the start of the path of
# a GitHub or raw GitHub URL of this repository, with or without its scheme, or as a relative path such as
# agents/rules.md or ../../AGENTS.md. A folder named agents inside another path, such as targets/codex/agents/, is
# not one.
AGENT_FILE_REFERENCE = re.compile(
    r"(?:(?:https?://)?(?:www\.)?github\.com/sima-neat/models/(?:blob|tree|edit|raw)/[^/\s]+/"
    r"|(?:https?://)?raw\.githubusercontent\.com/sima-neat/models/(?:refs/heads/)?[^/\s]+/)"
    r"(?:agents(?![\w.-])|AGENTS\.md)"
    r"|(?<![\w./-])(?:\.\./)*(?:agents/|AGENTS\.md)"
)
# The target of a Markdown link or image, inline as ](target) or in a reference definition as [label]: target.
MARKDOWN_TARGET = re.compile(r"\]\(\s*<?([^)\s>]+)|^ {0,3}\[[^]]+\]:\s*<?([^\s>]+)", re.MULTILINE)


def _documents() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "website" / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "agents").rglob("*.md")),
        *sorted((ROOT / "skills").rglob("*.md")),
        *sorted((ROOT / "artifact-browser").glob("*.md")),
    ]


def _skills() -> list[Path]:
    """Each skill folder: those for agents in this repository under agents/skills/, and customers' under skills/."""
    skills = [*ROOT.glob("agents/skills/*/SKILL.md"), *ROOT.glob("skills/*/SKILL.md")]
    return sorted(skill.parent for skill in skills)


def _customer_skill_folders() -> list[Path]:
    return sorted(folder for folder in ROOT.glob("skills/*") if folder.is_dir())


def _agent_skill_folders() -> list[Path]:
    return sorted(folder for folder in ROOT.glob("agents/skills/*") if folder.is_dir())


def _frontmatter(document: Path) -> dict:
    match = FRONTMATTER.match(document.read_text(encoding="utf-8"))
    try:
        fields = yaml.safe_load(match.group(1)) if match else None
    except yaml.YAMLError as error:
        raise AssertionError(f"{document.relative_to(ROOT)}: the frontmatter is not YAML: {error}") from error
    return fields if isinstance(fields, dict) else {}


def _prose(text: str) -> str:
    """The text outside fenced code blocks, where a link or heading is part of the page."""
    kept, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            kept.append(line)
    return "\n".join(kept)


def _anchors(document: Path) -> set[str]:
    """The heading anchors GitHub and the site generate for a Markdown file."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in _prose(document.read_text(encoding="utf-8")).splitlines():
        heading = re.match(r"#{1,6}\s+(.*)", line)
        if not heading:
            continue
        title = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading.group(1)).strip()
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        anchors.add(f"{slug}-{seen[slug]}" if slug in seen else slug)
        seen[slug] = seen.get(slug, 0) + 1
    return anchors


def _link_targets(document: Path) -> list[tuple[str, Path, str]]:
    """Each link into this repository as (written target, file, anchor)."""
    targets = []
    for written in MARKDOWN_LINK.findall(_prose(document.read_text(encoding="utf-8"))):
        target = written.split()[0]
        repository_link = MODELS_REPOSITORY_LINK.fullmatch(target)
        if repository_link:
            path, _, anchor = repository_link.group(1).partition("#")
            targets.append((target, ROOT / path, anchor))
        elif "://" not in target and not target.startswith("mailto:"):
            path, _, anchor = target.partition("#")
            targets.append((target, (document.parent / path).resolve() if path else document, anchor))
    return targets


def test_links_and_anchors_resolve_in_every_document() -> None:
    failures = []
    for document in _documents():
        for target, path, anchor in _link_targets(document):
            if not path.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}: no such file")
            elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
                failures.append(f"{document.relative_to(ROOT)} -> {target}: no such heading")
    assert not failures, "broken links:\n" + "\n".join(failures)


def test_contribution_path_routes_resolve() -> None:
    component = (ROOT / "website" / "src" / "components" / "ContributionPath.tsx").read_text(encoding="utf-8")
    routes = re.findall(r"guide: '([^']+)'", component)

    assert routes
    for route in routes:
        page, _, anchor = route.partition("#")
        document = ROOT / "docs" / f"{page}.md"
        assert document.is_file(), f"ContributionPath links to a missing page: {route}"
        assert not anchor or anchor in _anchors(document), f"ContributionPath links to a missing heading: {route}"


def test_site_pages_do_not_link_to_agent_or_maintainer_files() -> None:
    failures = []
    for page in sorted((ROOT / "docs").glob("*.md")):
        for target, path, _ in _link_targets(page):
            relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else target
            if relative.startswith(INTERNAL_TREES):
                failures.append(f"{page.relative_to(ROOT)} -> {target}")
    assert not failures, "docs/ is the public site and stands on its own:\n" + "\n".join(failures)


def test_agents_index_lists_every_docs_agents_and_skill_file() -> None:
    agents = ROOT / "AGENTS.md"
    indexed = {path for _, path, _ in _link_targets(agents)}
    pages = [page for page in sorted((ROOT / "docs").glob("*.md")) if page.name != "README.md"]
    missing = [
        page.relative_to(ROOT).as_posix()
        for page in [*pages, *sorted((ROOT / "agents").rglob("*.md")), *sorted(ROOT.glob("skills/*/SKILL.md"))]
        if page.resolve() not in indexed
    ]

    assert not missing, "add an index row to AGENTS.md for: " + ", ".join(missing)
    assert len(agents.read_text(encoding="utf-8").splitlines()) <= AGENTS_LINE_BUDGET


def test_skills_link_every_reference() -> None:
    unlinked = []
    for skill in _skills():
        linked = {path for _, path, _ in _link_targets(skill / "SKILL.md")}
        unlinked += [
            document.relative_to(ROOT).as_posix()
            for document in sorted(skill.rglob("*.md"))
            if document != skill / "SKILL.md" and document.resolve() not in linked
        ]
    assert not unlinked, "link each Markdown file from its skill's SKILL.md: " + ", ".join(unlinked)


def test_skills_declare_a_name_and_a_description() -> None:
    failures = []
    for skill in sorted([*(ROOT / "agents").rglob("SKILL.md"), *(ROOT / "skills").rglob("SKILL.md")]):
        fields = _frontmatter(skill)
        for field in ("name", "description"):
            if not isinstance(fields.get(field), str) or not fields[field].strip():
                failures.append(f"{skill.relative_to(ROOT)}: no {field} in its YAML frontmatter")
    assert not failures, "\n".join(failures)


def test_skills_are_named_after_their_folder() -> None:
    failures = []
    for folder in [*_customer_skill_folders(), *_agent_skill_folders()]:
        skill, name = folder / "SKILL.md", folder.relative_to(ROOT).as_posix()
        if not skill.is_file():
            failures.append(f"{name}: no SKILL.md")
        elif _frontmatter(skill).get("name") != folder.name:
            failures.append(f"{name}/SKILL.md: name must be {folder.name}")
    for folder in _customer_skill_folders():
        playbook = folder / "playbook.yml"
        playbook_fields = yaml.safe_load(playbook.read_text(encoding="utf-8")) if playbook.is_file() else None
        if not isinstance(playbook_fields, dict) or playbook_fields.get("id") != folder.name:
            failures.append(f"skills/{folder.name}: playbook.yml must exist with id {folder.name}")
    assert not failures, "\n".join(failures)


def _agent_file_references(text: str) -> list[str]:
    """Each place the text names agents/ or AGENTS.md of this repository."""
    return [match.group(0) for match in AGENT_FILE_REFERENCE.finditer(text)]


def _links_out_of(document: Path, folder: Path) -> list[str]:
    """Each relative Markdown link or image in the document, code blocks included, whose file is outside folder."""
    leaving = []
    for match in MARKDOWN_TARGET.finditer(document.read_text(encoding="utf-8")):
        target = match.group(1) or match.group(2)
        path = target.partition("#")[0]
        if path and "://" not in path and not path.startswith("mailto:"):
            if not (document.parent / path).resolve().is_relative_to(folder.resolve()):
                leaving.append(target)
    return leaving


def _text_files(folder: Path) -> list[tuple[Path, str]]:
    """Each file under folder that reads as UTF-8 text, with its text. A binary file names no path."""
    files = []
    for path in sorted(path for path in folder.rglob("*") if path.is_file()):
        try:
            files.append((path, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    return files


@pytest.mark.parametrize(
    ("text", "names_agent_files"),
    [
        ("https://github.com/sima-neat/models/blob/main/agents/rules.md", True),
        ("<https://github.com/sima-neat/models/blob/develop/AGENTS.md>", True),
        ("github.com/sima-neat/models/tree/main/agents", True),
        ("https://raw.githubusercontent.com/sima-neat/models/main/agents/rules.md", True),
        ("raw.githubusercontent.com/sima-neat/models/refs/heads/main/AGENTS.md", True),
        ("[rules](../../agents/rules.md)", True),
        ("Read AGENTS.md first.", True),
        ("https://github.com/sima-neat/models/blob/main/docs/datasets.md", False),
        ("https://github.com/sima-neat/models/blob/main/models/yolo26n/README.md", False),
        ("targets/codex/agents/openai.yaml", False),
        ("agents: [codex, claude]", False),
        ("A customer's agents run this skill.", False),
    ],
)
def test_agent_file_references_match_every_form(text: str, names_agent_files: bool) -> None:
    assert bool(_agent_file_references(text)) is names_agent_files


def test_links_out_of_a_skill_include_images_and_reference_definitions(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    (skill / "references").mkdir(parents=True)
    document = skill / "SKILL.md"
    document.write_text(
        "[inside](references/a.md#step) [web](https://example.com/x.md) [here](#steps)\n"
        "[up](../other.md) ![image](../../hero.png)\n"
        "[defined]: ../defined.md\n",
        encoding="utf-8",
    )

    leaving = _links_out_of(document, skill)

    assert leaving == ["../other.md", "../../hero.png", "../defined.md"]


def test_customer_skills_stand_alone() -> None:
    failures = []
    skills = ROOT / "skills"
    for document, text in _text_files(skills):
        name = document.relative_to(ROOT).as_posix()
        failures += [f"{name} names {reference}" for reference in _agent_file_references(text)]
        if document.suffix == ".md":
            parts = document.relative_to(skills).parts
            folder = skills / parts[0] if len(parts) > 1 else skills
            failures += [f"{name} -> {target}" for target in _links_out_of(document, folder)]
    assert not failures, "skills/ ships alone, so link only its own files and public URLs:\n" + "\n".join(failures)


def test_site_pages_open_with_a_title_and_an_introduction() -> None:
    for page in sorted((ROOT / "docs").glob("*.md")):
        lines = [line for line in _prose(page.read_text(encoding="utf-8")).splitlines() if line.strip()]
        lines = [line for line in lines if not line.startswith("import ")]
        assert lines[0].startswith("# "), f"{page.name} must start with its title"
        assert not lines[1].startswith("#"), f"{page.name} needs an introduction before its first section"


def test_site_code_blocks_name_their_language() -> None:
    failures = []
    for page in sorted((ROOT / "docs").glob("*.md")):
        fenced = False
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.lstrip().startswith("```"):
                continue
            if not fenced and line.strip() == "```":
                failures.append(f"{page.name}:{number}")
            fenced = not fenced
    assert not failures, "name the language of each code block: " + ", ".join(failures)


def test_docs_folder_holds_only_pages_and_assets() -> None:
    unexpected = [
        entry.name
        for entry in sorted((ROOT / "docs").iterdir())
        if not (entry.suffix == ".md" and entry.is_file()) and entry.name != "assets"
    ]

    assert not unexpected, "docs/ is the published site; move these elsewhere: " + ", ".join(unexpected)
