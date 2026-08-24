"""Static isolation contract for the RoboTTT package."""

from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_POLICY_ROOTS = {"DP", "DP_TTT", "DiT", "DiT_TTT"}


def imported_root(node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom) and node.module:
        return node.module.split(".", 1)[0]
    if isinstance(node, ast.Import) and node.names:
        return node.names[0].name.split(".", 1)[0]
    return None


def main() -> None:
    package = Path(__file__).resolve().parent
    violations: list[str] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            root = imported_root(node)
            if root in FORBIDDEN_POLICY_ROOTS:
                violations.append(f"{path.name}:{getattr(node, 'lineno', '?')} imports {root}")
    if violations:
        raise AssertionError("RoboTTT is not standalone:\n" + "\n".join(violations))
    print(
        {
            "standalone_policy_imports": True,
            "forbidden_policy_roots": sorted(FORBIDDEN_POLICY_ROOTS),
            "python_files_checked": len(list(package.glob("*.py"))),
        }
    )


if __name__ == "__main__":
    main()

