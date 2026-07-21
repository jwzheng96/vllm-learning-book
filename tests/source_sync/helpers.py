import json
import subprocess
from pathlib import Path


REPOSITORY = "https://github.com/vllm-project/vllm"
VALIDATED_AT = "2026-07-20T15:00:00Z"


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def init_git(path: Path) -> None:
    path.mkdir(parents=True)
    git(path, "init")
    git(path, "config", "user.name", "Source Sync Tests")
    git(path, "config", "user.email", "source-sync@example.invalid")


def commit_all(path: Path, message: str) -> str:
    git(path, "add", ".")
    git(path, "commit", "-m", message)
    return git(path, "rev-parse", "HEAD")


def create_source_repo(root: Path) -> tuple[Path, str, str]:
    source_repo = root / "upstream"
    init_git(source_repo)
    (source_repo / "vllm").mkdir()
    (source_repo / "vllm" / "example.py").write_text(
        "class Engine:\n"
        "    def run(self):\n"
        "        budget = 8\n"
        "        return budget\n",
        encoding="utf-8",
    )
    (source_repo / "csrc").mkdir()
    (source_repo / "csrc" / "example.cu").write_text(
        "const int block_size = 16;\n", encoding="utf-8"
    )
    baseline = commit_all(source_repo, "baseline")
    committed_at = git(source_repo, "show", "-s", "--format=%cI", baseline)
    return source_repo, baseline, committed_at


def version_block(commit: str, committed_at: str) -> str:
    return (
        "<!-- vllm-version:start -->\n"
        f"Validated vLLM: `{commit}`  \n"
        f"Upstream committed: `{committed_at}`  \n"
        f"Validated: `{VALIDATED_AT}`\n"
        "<!-- vllm-version:end -->"
    )


def write_lock(repo: Path, commit: str, committed_at: str) -> None:
    data = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "branch": "main",
        "commit": commit,
        "committed_at": committed_at,
        "validated_at": VALIDATED_AT,
    }
    (repo / "source.lock.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_tutorial_repo(root: Path) -> tuple[Path, str, str]:
    source_repo, source_sha, committed_at = create_source_repo(root)
    tutorial = root / "tutorial"
    init_git(tutorial)
    git(
        tutorial,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(source_repo),
        "vllm",
    )
    git(tutorial / "vllm", "config", "user.name", "Source Sync Tests")
    git(
        tutorial / "vllm",
        "config",
        "user.email",
        "source-sync@example.invalid",
    )
    chapter_dir = tutorial / "01-overview"
    chapter_dir.mkdir()
    source_url = (
        f"{REPOSITORY}/blob/{source_sha}/vllm/example.py#L3"
    )
    (chapter_dir / "01-intro.md").write_text(
        "# Introduction\n\n"
        '<!-- vllm-source: {"path":"vllm/example.py",'
        '"symbol":"Engine.run","anchor":"budget = 8","span":1} -->\n'
        f"[Engine budget]({source_url})\n",
        encoding="utf-8",
    )
    (tutorial / "README.md").write_text(
        "# Tutorial\n\n" + version_block(source_sha, committed_at) + "\n",
        encoding="utf-8",
    )
    (tutorial / "curriculum.toml").write_text(
        "[[chapter]]\n"
        'path = "01-overview/01-intro.md"\n'
        'title = "Introduction"\n'
        'level = "beginner"\n'
        'tracks = ["quickstart", "internals"]\n'
        'environments = ["no-gpu"]\n'
        'source_areas = ["vllm/v1/engine/**"]\n',
        encoding="utf-8",
    )
    (tutorial / "content-review.toml").write_text(
        "[[review]]\n"
        'path = "01-overview/01-intro.md"\n'
        'status = "pending"\n'
        'reviewed_commit = ""\n'
        'reviewed_at = ""\n'
        "source_contracts = false\n"
        "commands_checked = false\n"
        "metrics_checked = false\n"
        "diagrams_checked = false\n"
        "hardware_verified = []\n"
        "notes = []\n",
        encoding="utf-8",
    )
    write_lock(tutorial, source_sha, committed_at)
    commit_all(tutorial, "tutorial baseline")
    return tutorial, source_sha, committed_at
