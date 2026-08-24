import shutil
import subprocess
from pathlib import Path


def test_all_python_source_files_are_tracked_by_git() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    git_executable = shutil.which("git")
    assert git_executable is not None, "Git is required for repository checks"
    # The executable comes from the local path, not repository or user input.
    tracked = subprocess.run(  # noqa: S603
        [git_executable, "ls-files", "--", "src"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_python = {
        Path(line).as_posix()
        for line in tracked.stdout.splitlines()
        if line.endswith(".py")
    }
    source_python = {
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "src").rglob("*.py")
    }

    assert source_python <= tracked_python, (
        "Python source files are missing from Git: "
        f"{sorted(source_python - tracked_python)}"
    )
