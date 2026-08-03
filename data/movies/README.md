# Movie provenance

Every movie used by this project, where it came from, and its checksum. Verify
with `md5 <file>`; check a movie against a ROM with
`tasdata parse <file> --rom smb.nes`.

## Canonical

### `happylee_mars608-smb-warpless.fm2`  ← the one to use

| | |
| --- | --- |
| md5 | `97374c81541e9d3fc4881a4836c9af77` |
| source | TASVideos publication **3728** (`https://tasvideos.org/3728M`) |
| authors | HappyLee & mars608 |
| frames | 67,117 (18m36.8s @ 60.0988 Hz) |
| region | NTSC (`palFlag 0`) |
| needs ROM | `md5(prg+chr) = 8e3630186e35d477231bf8fd50e54cdd` — `Super Mario Bros. (JU) [!]` |
| status | **verified SYNCED** end to end under FCEUX 2.6.6 — all 32 levels, 1-1 → 8-4 |

`happylee_mars608-smb-warpless.fm2.zip` is the original download straight from
TASVideos (publications are served as a zip wrapping the `.fm2`). The parser
unwraps single-file zips transparently, so either path works and both parse
identically. Kept as the untouched provenance artifact.

> **Correction, recorded deliberately.** During earlier work this file was
> suspected of being corrupt — "missing most of its B presses", 44% input coverage
> against 99% for what was believed to be the published version. That was wrong.
> Downloading publication 3728 directly gives md5 `97374c81541e9d3fc4881a4836c9af77`,
> **byte-identical to this file**. The 44.1% input coverage and 20,478 `B` frames
> are simply what this run looks like: SMB does not need `B` held to maintain
> running speed once at maximum, and long stretches (level transitions, the 2-1
> vine, castle walks) take no input at all. The file was never quarantined. The
> comparison that produced the false alarm was against `userfile-638909616499952431`
> below, which is a *different author's* run that merely happens to share the exact
> frame count.

## Secondary test movies

### `userfile-638909616499952431-smb-warpless.fm2`

| | |
| --- | --- |
| md5 | `d5c696b7b8522151b6f51bef41c1eba8` |
| source | TASVideos **user file** 638909616499952431 (not a publication) |
| frames | 67,117 — coincidentally identical to publication 3728 |
| differs from 3728 | on 56,448 of 67,117 port-0 frames; presses Start at frame 33 vs 41; holds `B` on 66,329 frames vs 20,478 |
| needs ROM | same NTSC dump as above |

A separate warpless attempt by a different author. Useful as a second NTSC test
case, and as the cautionary example: identical length and game do **not** imply
the same input log. Do not treat it as a copy of the publication.

### `638820107290643872__pal_warpless.fm2`

| | |
| --- | --- |
| md5 | `c8b93dbf8fcc8fe414f4a0ed6cf19822` |
| source | TASVideos user file 638820107290643872 |
| region | **PAL** (`palFlag 1`), `Super Mario Bros. (Europe) (Rev 0A)` |
| needs ROM | `md5(prg+chr) = ba39dde63ab209b1bc751e0535e72b18` — the ROM `gym-super-mario-bros` bundles |

Retained purely as the **nes-py PAL regression pair**: it is the only movie whose
ROM fingerprint matches the pip-installable ROM, so it exercises the ROM-matching
path and the nes-py backend without needing a ROM on disk. It is not training
data — nes-py has no PAL timing, and it desyncs at frame 877.

## Nothing here is quarantined

There is no `corrupt/` directory: no movie in this project has been shown to be
damaged. If one ever is, quarantine it here with its md5, the evidence, and the
command that produced that evidence.
