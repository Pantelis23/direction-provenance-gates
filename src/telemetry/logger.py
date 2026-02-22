import json
from pathlib import Path
from typing import Any, Dict, Optional


class TelemetryLogger:
    def __init__(self, path: Path):
        self.path = path
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
