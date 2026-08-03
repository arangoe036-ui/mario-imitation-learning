"""Fetch TAS movies from tasvideos.org.

Movies are the input data for this pipeline, and hunting them down by hand is
tedious: publications and user files live behind different endpoints, and user
files are served gzip-wrapped.  This module handles both and sniffs what it got.

Only movie files are downloaded.  ROMs are not distributed by TASVideos and are
not fetched here -- supply your own (see the README).
"""

from __future__ import annotations

import gzip
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .formats import MovieFormat, sniff

API = "https://tasvideos.org/api/v1"
USER_AGENT = "tasdata/0.1 (TAS imitation-learning data pipeline)"
_TIMEOUT = 90


@dataclass
class FetchedMovie:
    """One downloaded movie file."""

    path: Path
    source: str
    format: MovieFormat
    size: int
    title: str = ""

    def line(self) -> str:
        return f"{self.path.name}  [{self.format.value}] {self.size} B  <- {self.source}"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def _get_with_filename(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        disposition = resp.headers.get("Content-Disposition", "")
        match = re.search(r'filename="?([^";]+)', disposition)
        return resp.read(), (match.group(1) if match else "")


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("_") or "movie"


def _write(data: bytes, dest: Path, source: str, title: str = "") -> FetchedMovie:
    """Write bytes to ``dest``, stripping an outer gzip layer if present."""
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except OSError:
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    result = sniff(dest)
    return FetchedMovie(dest, source, result.format, len(data), title)


def search_games(name: str, system: str = "NES") -> list[dict]:
    """Look up games by display name. The API caps results, so filter locally."""
    data = json.loads(_get(f"{API}/games?systemCodes={system}&limit=1000"))
    needle = name.lower()
    return [g for g in data if needle in str(g.get("displayName", "")).lower()]


def list_publications(game_id: int) -> list[dict]:
    """Published movies for a game id."""
    return json.loads(_get(f"{API}/publications?gameIds={game_id}&limit=500"))


def fetch_publication(pub_id: int, out_dir: Path | str) -> FetchedMovie:
    """Download a publication's movie file by publication id."""
    out_dir = Path(out_dir)
    url = f"https://tasvideos.org/{pub_id}M?handler=Download"
    data, name = _get_with_filename(url)
    dest = out_dir / _safe_name(name or f"pub{pub_id}.bk2")
    return _write(data, dest, url)


def user_file_ids(game_id: int) -> list[str]:
    """Scrape the user-file listing for a game.

    There is no public API for user files, so this parses the HTML listing. It is
    deliberately forgiving: a layout change yields an empty list, not a crash.
    """
    html = _get(f"https://tasvideos.org/UserFiles/Game/{game_id}").decode(
        "utf-8", errors="replace"
    )
    return sorted(set(re.findall(r"/UserFiles/Info/(\d+)\?handler=Download", html)))


def fetch_user_file(file_id: str, out_dir: Path | str) -> FetchedMovie:
    """Download one user file by its numeric id."""
    out_dir = Path(out_dir)
    url = f"https://tasvideos.org/UserFiles/Info/{file_id}?handler=Download"
    data, name = _get_with_filename(url)
    dest = out_dir / _safe_name(f"{file_id}__{name or 'userfile'}")
    return _write(data, dest, url)


def fetch_game_movies(
    game_id: int,
    out_dir: Path | str,
    *,
    only_bk2: bool = True,
    limit: int | None = None,
    include_publications: bool = True,
    on_progress=None,
) -> list[FetchedMovie]:
    """Download every movie TASVideos has for a game.

    Args:
        only_bk2: discard anything that is not a BizHawk ``.bk2`` after sniffing.
        limit: stop after this many *kept* movies.
        include_publications: also fetch published movies, not just user files.
    """
    out_dir = Path(out_dir)
    kept: list[FetchedMovie] = []
    sources: list[tuple[str, str]] = [("user", i) for i in user_file_ids(game_id)]
    if include_publications:
        for pub in list_publications(game_id):
            sources.append(("pub", str(pub["id"])))

    for kind, ident in sources:
        if limit is not None and len(kept) >= limit:
            break
        try:
            movie = (
                fetch_user_file(ident, out_dir)
                if kind == "user"
                else fetch_publication(int(ident), out_dir)
            )
        except Exception as exc:  # network flake, dead id, corrupt payload

            if on_progress:
                on_progress(f"skip {kind} {ident}: {type(exc).__name__}: {exc}")
            continue
        if only_bk2 and movie.format is not MovieFormat.BK2:
            movie.path.unlink(missing_ok=True)
            if on_progress:
                on_progress(f"drop {movie.path.name}: {movie.format.value}, not bk2")
            continue
        kept.append(movie)
        if on_progress:
            on_progress(f"kept {movie.line()}")
    return kept
