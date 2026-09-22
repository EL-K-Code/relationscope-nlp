"""Download the three FewRel files required by RelationScope."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

BASE = "https://raw.githubusercontent.com/thunlp/FewRel/master/data"
FILES = ("train_wiki.json", "val_wiki.json", "pid2name.json")
TARGET = Path("data/raw/fewrel")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        for name in FILES:
            output = TARGET / name
            if output.exists():
                print(f"exists: {output}  sha256={sha256(output)}")
                continue

            url = f"{BASE}/{name}"
            print(f"downloading {url}")
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with output.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            print(f"saved: {output}  sha256={sha256(output)}")


if __name__ == "__main__":
    main()
