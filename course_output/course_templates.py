"""
course_output/course_templates.py

Resolves the bundled blank-course template for a given game_version +
theme -- the source for automatic course/ baseline provisioning (see
PGA2k_gen.py's _ensure_course_baseline). Replaces the old manual
"browse to a .course file, click Ingest" workflow: a blank course is
now a bundled app asset, not something the user supplies per project.

Templates live at <repo root>/templates/{game_version}_{theme}.course,
one blank per (version, theme) combination actually in use (e.g.
templates/2019_rustic.course). Adding support for a new version/theme
combination is a matter of dropping a new file here, not a code
change.
"""
from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR_NAME = "templates"


def resolve_course_template(script_dir: Path, game_version: str, theme: str) -> Path:
    """
    <script_dir>/templates/{game_version}_{theme}.course, where theme
    is lowercased/stripped into the filename slug. Raises
    FileNotFoundError (not this project's StepError, to avoid a
    circular import back into PGA2k_gen.py -- callers there wrap it)
    if no such file exists.
    """
    theme_slug = theme.strip().lower()
    path = Path(script_dir) / TEMPLATES_DIR_NAME / f"{game_version}_{theme_slug}.course"
    if not path.exists():
        raise FileNotFoundError(
            f"No bundled template at {path} -- add a blank .course file there for "
            f"game_version={game_version!r} theme={theme_slug!r} (see "
            f"templates/2019_rustic.course for the naming convention)."
        )
    return path
