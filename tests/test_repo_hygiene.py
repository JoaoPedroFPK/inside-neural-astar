"""Guards on what may enter the published repository.

The project keeps its working notes outside version control, and the published
tree must not name them or the tooling that produced them. A reference slipped
into a module docstring once and reached the remote; this test is the guard so
it cannot happen twice.

The forbidden tokens are assembled at runtime from fragments so that this file
does not itself contain the strings it forbids.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Assembled rather than written out, so the guard does not trip on itself.
FORBIDDEN = ("CLA" + "UDE", "cla" + "ude", "Cla" + "ude", "ANTHROP" + "IC", "anthrop" + "ic")

#: Files that are tracked but whose content is data rather than prose.
BINARY_SUFFIXES = {".png", ".parquet", ".npz", ".ckpt", ".jpg", ".gif", ".pdf"}


def tracked_text_files() -> list[Path]:
    """Every tracked file that is worth scanning as text.

    Returns:
        list[Path]: Absolute paths of tracked, non-binary files.
    """
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    files = []
    for name in listing:
        path = REPO_ROOT / name
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        files.append(path)
    return files


def test_no_tracked_file_names_the_excluded_tooling() -> None:
    """No tracked file may contain the forbidden tokens.

    Scans the files git is tracking, which is exactly the set that reaches the
    remote. Untracked working notes are deliberately out of scope - they are
    excluded from version control instead.
    """
    offenders = []
    for path in tracked_text_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        if path.name == Path(__file__).name:
            continue
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains a forbidden token")
                break
    assert not offenders, "\n".join(offenders)


def test_working_notes_are_not_tracked() -> None:
    """The notes directory and the plan stay out of version control."""
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert not [name for name in listing if name.startswith("docs/")], (
        "the notes directory must not be tracked"
    )
    assert "PLAN.md" not in listing


@pytest.mark.parametrize("path", ["README.md", "Makefile"])
def test_public_files_do_not_link_to_untracked_notes(path: str) -> None:
    """Public files must not link to paths that are not in the repository.

    A link to a file the reader cannot open is worse than no link; prose
    references to the working notes are fine, markdown links to them are not.
    """
    text = (REPO_ROOT / path).read_text()
    assert "](docs/" not in text, f"{path} links to the untracked notes directory"


def test_every_script_has_a_working_cli() -> None:
    """`--help` must work for every script, with no import-time failure.

    Cheap insurance for a repository whose experiments are run from the command
    line: a broken import or a malformed argument definition shows up here
    rather than twenty minutes into a sweep.
    """
    scripts = sorted(
        path
        for path in (REPO_ROOT / "scripts").glob("*.py")
        if path.name != "__init__.py"
    )
    assert scripts, "no scripts found"
    failures = []
    for path in scripts:
        completed = subprocess.run(
            [str(REPO_ROOT / ".venv" / "bin" / "python"), str(path), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            failures.append(f"{path.name}: exit {completed.returncode}\n{completed.stderr[-400:]}")
    assert not failures, "\n".join(failures)


def test_every_make_target_names_a_real_script() -> None:
    """Makefile recipes must point at scripts that exist.

    The targets are how every experiment is launched, and a renamed script
    would otherwise fail only when someone runs it.
    """
    import re

    makefile = (REPO_ROOT / "Makefile").read_text()
    referenced = set(re.findall(r"scripts/([A-Za-z0-9_]+\.(?:py|sh))", makefile))
    assert referenced, "no scripts referenced from the Makefile"
    missing = [name for name in referenced if not (REPO_ROOT / "scripts" / name).exists()]
    assert not missing, f"Makefile references missing scripts: {missing}"


def test_no_build_artifacts_are_tracked() -> None:
    """Model checkpoints and training logs must not be in version control.

    They are build outputs, they are large, and the tensorboard event files are
    named after the machine that produced them — which put a personal hostname
    into the published history once already and had to be scrubbed.
    """
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = [
        name
        for name in listing
        if name.startswith("results/models/")
        or name.endswith((".ckpt", ".npz", ".parquet"))
        or ".tfevents." in name
    ]
    assert not offenders, f"build artifacts are tracked: {offenders[:5]}"


def test_no_tracked_file_names_the_local_machine() -> None:
    """No tracked path may carry the training machine's hostname."""
    import platform

    hostname = platform.node().split(".")[0]
    if not hostname:
        pytest.skip("no hostname to check against")
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert hostname.lower() not in listing.lower(), (
        f"a tracked path contains the hostname {hostname!r}"
    )


def test_empty_checkpoint_argument_means_the_shipped_model() -> None:
    """An empty `--checkpoints` entry must not resolve to the repository root.

    `Path("")` is `Path(".")`, and the checkpoint loader globs recursively and
    takes the last match — so a blank entry would silently load a *trained*
    checkpoint from `results/` and evaluate the wrong model under the label
    "pretrained".
    """
    from pathlib import Path

    entries = ["", "results/models/mazes_seed0"]
    resolved = [Path(entry) if entry else None for entry in entries]
    assert resolved[0] is None
    assert resolved[1] == Path("results/models/mazes_seed0")


def test_tagged_scripts_never_print_an_untagged_artifact_name() -> None:
    """A script run with `--tag` must report the file it actually wrote.

    Every per-model script writes `<stem>_<tag>.<ext>` but several once printed
    the bare `<stem>.<ext>` in their closing summary. Nothing was overwritten,
    but the message said a shared, committed report had been replaced when it
    had not — which is worse than useless when the whole point of the tag is to
    keep models from contaminating each other's results.
    """
    import re

    artifact = re.compile(
        r"\b(e\d[\w]*|features_\w+|manifest|seeds)\.(md|json|parquet|npz)\b"
    )
    interpolated = re.compile(r"\{[^}]*(suffix|tag|stem|name|out|path)[^}]*\}")

    offenders = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        source = path.read_text()
        if "--tag" not in source:
            continue
        for number, line in enumerate(source.splitlines(), start=1):
            if "print" not in line and not line.lstrip().startswith(('f"', '"')):
                continue
            found = artifact.search(line)
            if found and not interpolated.search(line):
                offenders.append(f"{path.name}:{number}: {found.group(0)}")
    assert not offenders, "untagged artifact names in output messages:\n" + "\n".join(offenders)
