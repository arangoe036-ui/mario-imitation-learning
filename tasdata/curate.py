"""Candidate selection: decide which movies are worth capturing.

Selection rules, in order of authority:

1. **ROM identity.** A movie whose ``romChecksum`` does not match the supplied ROM
   is rejected outright. NTSC only; PAL movies are rejected too, since FCEUX would
   need different timing and the resulting frames would not be comparable.
2. **Category exclusions.** Glitch and constrained-movement categories are
   excluded: their inputs either drive memory corruption or are artificially
   restricted, so they teach the wrong thing.
3. **Coverage priority.** Warpless categories visit all 32 levels; warps runs see
   four. Warpless is therefore worth far more per megabyte and is ranked first.
4. **Obsoletion chains.** Walking backwards from each current record picks up older,
   less glitch-heavy versions of the same category -- more behavioural variety for
   the same game.

Publications are preferred over user files: they are vetted, categorised by
TASVideos, and carry authorship. User files fill the remainder of the target.

Old publications are often ``.fcm`` (pre-2008 FCEU) or ``.fmv`` (Famtasia).
``.fcm`` is converted to ``.fm2`` with FCEUX's own ``--fcmconvert``, which
preserves the ROM checksum and syncs. ``.fmv`` is a different emulator's format
that FCEUX cannot play, so those are skipped.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .fceux_backend import find_fceux
from .formats import MovieFormat, sniff
from .movie import Movie, parse_movie
from .rom import NesRom, load_rom
from .tasvideos import USER_AGENT, _safe_name, fetch_publication, fetch_user_file, user_file_ids

API = "https://tasvideos.org/api/v1"

#: Categories to exclude, as regexes matched against the branch/goal or filename.
#: Each is either memory corruption (the inputs are an exploit payload, not play)
#: or artificially constrained movement (deliberately not how the game is played).
EXCLUDED_CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"game\W*end\W*glitch|\bgeg\b", "game-end-glitch"),
    (r"arbitrary\W*code|\bace\b", "arbitrary-code-execution"),
    (r"minimum\W*a\W*press|min\W*a\W*press", "minimum-A-presses"),
    (r"minimum\W*press|min\W*press|minpress", "minimum-presses"),
    (r"walkathon|walk\W*a\W*thon", "walkathon"),
    (r"maximum\W*score|max\W*score|maxscore", "maximum-score"),
    (r"maximum\W*coin|max\W*coin|maxcoin", "maximum-coins"),
)

#: Coverage rank: lower sorts first. Warpless visits 32 levels, warps only 4.
CATEGORY_RANK: tuple[tuple[str, int, str], ...] = (
    (r"warpless", 0, "warpless"),
    (r"all\W*items", 1, "all-items"),
    (r"no\W*glitch|glitchless", 2, "warpless-glitchless"),
    (r"warp", 5, "warps"),
)

#: Levels each category visits, for the value-per-megabyte column. Measured, not
#: assumed: the warps route is 1-1, 1-2, 4-1, 4-2, 8-1, 8-2, 8-3, 8-4 -- eight
#: levels across three worlds, not four levels.
CATEGORY_LEVELS = {
    "warpless": 32,
    "all-items": 32,
    "warpless-glitchless": 32,
    "warps": 8,
    "unknown": 0,
}

#: Bytes written per emulated frame by ``tasdata run`` at 84x84, every frame:
#: 84*84 image + 13 int32 trace columns + 1 action byte + 13 button bools.
BYTES_PER_FRAME_84 = 84 * 84 + 13 * 4 + 1 + 13


def estimate_bytes(n_frames: int, observation_shape: tuple[int, int] = (84, 84)) -> int:
    """Predicted on-disk size of a captured run."""
    height, width = observation_shape
    per_frame = height * width + 13 * 4 + 1 + 13
    return n_frames * per_frame


def excluded_reason(text: str) -> str | None:
    """Return the excluded-category name if ``text`` names one."""
    low = (text or "").lower()
    for pattern, name in EXCLUDED_CATEGORY_PATTERNS:
        if re.search(pattern, low):
            return name
    return None


def classify(text: str) -> str:
    """Map a branch/goal string or filename to a coverage category."""
    low = (text or "").lower()
    for pattern, _rank, name in CATEGORY_RANK:
        if re.search(pattern, low):
            return name
    return "unknown"


def category_rank(category: str) -> int:
    for pattern, rank, name in CATEGORY_RANK:
        if name == category:
            return rank
    return 9


@dataclass
class Candidate:
    """One movie that could be captured."""

    source: str                 # "publication" | "userfile"
    source_id: str
    label: str                  # human-readable provenance
    path: str                   # local movie file, post-conversion
    category: str
    authors: str
    n_frames: int
    movie_format: str
    pal: bool
    rom_ok: bool
    est_bytes: int
    #: Publication this one was obsoleted by, i.e. its successor in the chain.
    obsoleted_by: str | None = None
    #: Chain key grouping successive versions of the same category.
    chain: str = ""
    #: Position within the chain, 0 = current record, larger = older.
    chain_position: int = 0
    converted_from: str | None = None
    notes: list[str] = field(default_factory=list)
    rejected: str | None = None

    @property
    def levels(self) -> int:
        return CATEGORY_LEVELS.get(self.category, 0)

    def row(self) -> str:
        disk = self.est_bytes / (1 << 20)
        return (
            f"{self.label:26s} {self.category:20s} {self.authors[:26]:26s} "
            f"{self.n_frames:7d}f {self.levels:3d}L {disk:7.1f} MiB"
        )


def _get_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def fetch_publication_index(game_id: int) -> list[dict]:
    """All publications for a game, obsoleted ones included."""
    return _get_json(
        f"{API}/publications?gameIds={game_id}&limit=500&showObsoleted=true"
    )


def convert_fcm(path: Path, binary: str | Path = "fceux") -> Path:
    """Convert a legacy ``.fcm`` to ``.fm2`` using FCEUX's own converter.

    Writes alongside the input with an ``.fm2`` suffix and returns that path.

    Two quirks are handled here. The path is resolved to absolute because the
    converter interprets the argument relative to the process working directory,
    and FCEUX 2.6.6 *segfaults on exit* after a successful conversion -- so success
    is judged by the output file existing, never by the return code.
    """
    fceux = find_fceux(binary)
    path = Path(path).resolve()
    out = path.with_suffix(".fm2")
    if out.exists():
        return out
    proc = subprocess.run(
        [str(fceux), "--fcmconvert", str(path)],
        capture_output=True,
        timeout=300,
    )
    if not out.exists():
        tail = (proc.stderr or b"").decode(errors="replace")[-200:]
        raise RuntimeError(
            f"fcmconvert produced no output for {path.name} "
            f"(exit {proc.returncode}): {tail}"
        )
    return out


def _prepare(path: Path, binary: str | Path) -> tuple[Path, str | None]:
    """Return a parseable movie path, converting a legacy ``.fcm`` if needed.

    Dispatches on the *sniffed* format, not the extension: TASVideos serves
    publications zipped, so an old FCEU movie arrives as ``foo.fcm.zip`` and an
    extension check would miss it. FCEUX's converter also cannot read a zip, so the
    inner bytes are written out first.
    """
    result = sniff(path)
    if result.format is MovieFormat.FM2:
        # TASVideos serves publications zipped. Unwrap so the shortlist points at a
        # plain .fm2: FCEUX cannot read containers and would silently play nothing.
        if result.gzipped or result.inner_name:
            name = result.inner_name or path.name
            for wrapper in (".zip", ".gz"):
                if name.lower().endswith(wrapper):
                    name = name[: -len(wrapper)]
            if not name.lower().endswith(".fm2"):
                name += ".fm2"
            plain = path.parent / _safe_name(name)
            plain.write_bytes(result.data)
            return plain, path.name
        return path, None
    if result.format is not MovieFormat.FCM:
        return path, None
    name = path.name
    for wrapper in (".zip", ".gz"):
        if name.lower().endswith(wrapper):
            name = name[: -len(wrapper)]
    if not name.lower().endswith(".fcm"):
        name += ".fcm"
    fcm_path = path.parent / name
    fcm_path.write_bytes(result.data)
    return convert_fcm(fcm_path, binary), path.name


def _examine(
    path: Path,
    rom: NesRom,
    *,
    source: str,
    source_id: str,
    label: str,
    category: str,
    authors: str,
    obsoleted_by: str | None = None,
    converted_from: str | None = None,
    observation_shape: tuple[int, int] = (84, 84),
) -> Candidate:
    """Parse a downloaded movie and build a (possibly rejected) candidate."""
    notes: list[str] = []
    try:
        movie: Movie = parse_movie(path)
    except Exception as exc:
        return Candidate(
            source, source_id, label, str(path), category, authors, 0, "?", False,
            False, 0, obsoleted_by, converted_from=converted_from,
            rejected=f"parse failed: {type(exc).__name__}: {exc}"[:160],
        )

    check = movie.verify_rom(rom)
    rom_ok = check.ok
    est = estimate_bytes(movie.n_frames, observation_shape)
    cand = Candidate(
        source=source,
        source_id=source_id,
        label=label,
        path=str(path),
        category=category if category != "unknown" else classify(path.name),
        authors=authors or movie.author,
        n_frames=movie.n_frames,
        movie_format=movie.format.value,
        pal=movie.pal,
        rom_ok=rom_ok,
        est_bytes=est,
        obsoleted_by=obsoleted_by,
        converted_from=converted_from,
        notes=notes + movie.notes[:2],
    )
    if movie.format is not MovieFormat.FM2:
        cand.rejected = f"not an fm2 ({movie.format.value})"
    elif movie.pal:
        cand.rejected = "PAL movie (NTSC only)"
    elif not rom_ok:
        cand.rejected = f"ROM mismatch ({check.expected or 'no checksum'})"
    elif movie.savestate_anchored:
        cand.rejected = "anchored to a savestate"
    elif movie.n_frames < 3000:
        cand.rejected = f"too short to be a full run ({movie.n_frames} frames)"
    return cand


def gather_publications(
    game_id: int,
    rom: NesRom,
    pool: Path,
    *,
    binary: str | Path = "fceux",
    observation_shape: tuple[int, int] = (84, 84),
    on_progress=None,
) -> list[Candidate]:
    """Download and examine every non-excluded publication, obsoleted included."""
    pool.mkdir(parents=True, exist_ok=True)
    index = fetch_publication_index(game_id)
    # Chain successors: pub -> the pub that obsoleted it.
    out: list[Candidate] = []
    for pub in sorted(index, key=lambda p: p["frames"]):
        branch = str(pub.get("branch") or pub.get("goal") or "")
        category = classify(branch)
        excluded = excluded_reason(branch) or excluded_reason(pub["movieFileName"])
        label = f"pub {pub['id']}"
        authors = ",".join(pub.get("authors") or [])
        if excluded:
            out.append(
                Candidate(
                    "publication", str(pub["id"]), label, "", category, authors,
                    pub["frames"], pub["movieFileName"].rsplit(".", 1)[-1], False,
                    False, 0, rejected=f"excluded category: {excluded}",
                )
            )
            continue
        ext = pub["movieFileName"].rsplit(".", 1)[-1].lower()
        if ext in ("fmv", "nmv", "smv"):
            out.append(
                Candidate(
                    "publication", str(pub["id"]), label, "", category, authors,
                    pub["frames"], ext, False, False, 0,
                    rejected=f".{ext} is a foreign emulator format FCEUX cannot play",
                )
            )
            continue
        try:
            fetched = fetch_publication(int(pub["id"]), pool)
            path, converted_from = _prepare(fetched.path, binary)
        except Exception as exc:
            out.append(
                Candidate(
                    "publication", str(pub["id"]), label, "", category, authors,
                    pub["frames"], ext, False, False, 0,
                    rejected=f"download/convert failed: {type(exc).__name__}: {exc}"[:140],
                )
            )
            continue
        cand = _examine(
            path, rom, source="publication", source_id=str(pub["id"]), label=label,
            category=category, authors=authors,
            obsoleted_by=str(pub["obsoletedById"]) if pub.get("obsoletedById") else None,
            converted_from=converted_from, observation_shape=observation_shape,
        )
        out.append(cand)
        if on_progress:
            on_progress(f"{label}: {cand.rejected or 'ok'} ({cand.n_frames}f)")
    _assign_chains(out)
    return out


def _assign_chains(candidates: list[Candidate]) -> None:
    """Group publications into obsoletion chains and number them oldest-last."""
    by_id = {c.source_id: c for c in candidates if c.source == "publication"}
    for cand in by_id.values():
        # Walk forward to the newest publication in this chain; that id names it.
        seen, node = set(), cand
        while node.obsoleted_by and node.obsoleted_by in by_id:
            if node.obsoleted_by in seen:
                break
            seen.add(node.obsoleted_by)
            node = by_id[node.obsoleted_by]
        cand.chain = f"{cand.category}/{node.source_id}"
        cand.chain_position = len(seen)


def gather_userfiles(
    game_id: int,
    rom: NesRom,
    pool: Path,
    *,
    limit: int | None = None,
    observation_shape: tuple[int, int] = (84, 84),
    on_progress=None,
) -> list[Candidate]:
    """Download and examine TASVideos user files for a game."""
    pool.mkdir(parents=True, exist_ok=True)
    ids = user_file_ids(game_id)
    out: list[Candidate] = []
    for i, uid in enumerate(ids):
        if limit is not None and len(out) >= limit:
            break
        try:
            fetched = fetch_user_file(uid, pool)
        except Exception as exc:
            continue
        name = fetched.path.name
        excluded = excluded_reason(name)
        label = f"user {uid[:10]}"
        if excluded:
            fetched.path.unlink(missing_ok=True)
            out.append(
                Candidate(
                    "userfile", uid, label, "", classify(name), "", 0, "?", False,
                    False, 0, rejected=f"excluded category: {excluded}",
                )
            )
            continue
        cand = _examine(
            fetched.path, rom, source="userfile", source_id=uid, label=label,
            category=classify(name), authors="", observation_shape=observation_shape,
        )
        if cand.rejected:
            Path(cand.path).unlink(missing_ok=True) if cand.path else None
        out.append(cand)
        if on_progress and i % 25 == 0:
            on_progress(f"userfiles: {i}/{len(ids)} examined, {sum(1 for c in out if not c.rejected)} usable")
    return out


def select(
    candidates: list[Candidate],
    *,
    target: int = 40,
    max_low_coverage: int | None = None,
    low_coverage_levels: int = 16,
) -> tuple[list[Candidate], list[Candidate]]:
    """Rank usable candidates and take up to ``target`` of them.

    Ordering: coverage category first (warpless before warps), then publications
    before user files, then longest first within a chain position -- older chain
    members are slower and therefore longer, and they are the point of walking the
    chain backwards.

    Args:
        max_low_coverage: cap on runs visiting fewer than ``low_coverage_levels``
            levels -- i.e. less than half the game. Warps runs all cover the same
            eight levels, so after a handful they are near-duplicates of each
            other; the default caps them at a quarter of the target rather than
            letting them fill every slot left over once the warpless supply runs
            out.
    """
    if max_low_coverage is None:
        max_low_coverage = max(1, target // 4)
    usable = [c for c in candidates if not c.rejected]
    rejected = [c for c in candidates if c.rejected]

    # Drop byte-identical movies, keyed on content hash. Frame count alone is not
    # enough (two different warps runs are both ~17,900 frames) and a size-based key
    # silently collapses distinct runs whenever a file cannot be stat'd.
    seen: dict[str, Candidate] = {}
    deduped: list[Candidate] = []
    for c in sorted(usable, key=lambda c: (c.source != "publication", c.label)):
        try:
            digest = hashlib.md5(Path(c.path).read_bytes()).hexdigest()
        except OSError:
            digest = ""  # unreadable: keep it, let the capture stage fail loudly
        if digest and digest in seen:
            c.rejected = f"byte-identical to {seen[digest].label}"
            rejected.append(c)
            continue
        if digest:
            seen[digest] = c
        deduped.append(c)

    deduped.sort(
        key=lambda c: (
            category_rank(c.category),
            c.source != "publication",
            c.chain_position,
            -c.n_frames,
        )
    )
    selected: list[Candidate] = []
    low_taken = 0
    for c in deduped:
        if len(selected) >= target:
            c.rejected = "beyond target count"
            rejected.append(c)
            continue
        if c.levels < low_coverage_levels:
            if low_taken >= max_low_coverage:
                c.rejected = (
                    f"low-coverage cap reached ({max_low_coverage} runs of "
                    f"<{low_coverage_levels} levels)"
                )
                rejected.append(c)
                continue
            low_taken += 1
        selected.append(c)
    return selected, rejected


def write_plan(path: Path, selected: list[Candidate], rejected: list[Candidate]) -> Path:
    """Persist the shortlist so a capture run is reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected": [asdict(c) for c in selected],
        "rejected": [asdict(c) for c in rejected],
        "totals": {
            "n_runs": len(selected),
            "n_frames": sum(c.n_frames for c in selected),
            "est_bytes": sum(c.est_bytes for c in selected),
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def apply_measurements(plan_path: Path, measurements_path: Path) -> tuple[int, int]:
    """Replace declared categories with measured ones in a shortlist.

    Filename-derived categories are guesses; the RAM trace is ground truth. A movie
    called "...Warpless_TAS.fm2" that visits eight levels in five minutes is a warps
    run, and the manifest must say so or every downstream subset is wrong.

    Returns ``(n_relabelled, n_desynced)``.
    """
    plan = json.loads(Path(plan_path).read_text())
    measured = {m["label"]: m for m in json.loads(Path(measurements_path).read_text())}
    relabelled = desynced = 0
    for entry in plan["selected"]:
        m = measured.get(entry["label"])
        if not m:
            continue
        declared = entry["category"]
        route = m["route"]
        # Preserve a meaningful qualifier (e.g. glitchless) while fixing the route.
        qualifier = "-glitchless" if "glitchless" in declared else ""
        measured_category = route + (qualifier if route in ("warpless", "warps") else "")
        if declared == "all-items" and route == "warpless":
            measured_category = "all-items"  # all-items *is* a 32-level route
        entry["declared_category"] = declared
        entry["measured_levels"] = m["measured_levels"]
        entry["measured_route"] = route
        entry["furthest_level"] = m["furthest"]
        entry["premeasured_synced"] = m["synced"]
        if measured_category != declared:
            entry["category"] = measured_category
            relabelled += 1
        if not m["synced"]:
            desynced += 1
    Path(plan_path).write_text(json.dumps(plan, indent=2))
    return relabelled, desynced


def load_plan(path: Path) -> list[Candidate]:
    """Load a shortlist, ignoring any measurement fields added afterwards."""
    data = json.loads(Path(path).read_text())
    fields = set(Candidate.__dataclass_fields__)
    return [Candidate(**{k: v for k, v in c.items() if k in fields}) for c in data["selected"]]
