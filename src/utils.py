"""Common utility functions for the ESConv PoC."""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """Save data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file (JSON Lines)."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str):
    """Save data to JSONL file (JSON Lines)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def print_stats(title: str, stats: Dict[str, Any]):
    """Pretty print statistics."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"{'=' * 60}\n")
