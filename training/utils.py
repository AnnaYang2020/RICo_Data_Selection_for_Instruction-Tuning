"""Small JSON helpers used by the training scripts."""

import json
from pathlib import Path
from typing import Any, Union


def jdump(obj: Any, path: Union[str, Path], indent: int = 4) -> None:
    """Write a JSON-serializable object to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=indent, ensure_ascii=False)


def jload(path: Union[str, Path]) -> Any:
    """Read a JSON document from disk."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)
