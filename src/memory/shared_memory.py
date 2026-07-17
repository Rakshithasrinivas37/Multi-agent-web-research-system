"""Simple JSON-backed shared memory for agent outputs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union


class SharedMemory:
    """Stores agent outputs in one JSON file."""

    def __init__(self, path: Union[str, Path] = "data/shared_memory.json") -> None:
        self.path = Path(path)
        self.data = self.load()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def write_agent_output(self, agent_name: str, output: dict[str, Any]) -> None:
        self.data[agent_name] = {
            **output,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def read_agent_output(self, agent_name: str) -> dict[str, Any]:
        value = self.data.get(agent_name, {})
        return value if isinstance(value, dict) else {}
