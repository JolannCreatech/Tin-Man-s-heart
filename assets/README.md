# Local assets

`scripts/link_assets.sh` creates `assets/magenta-rt-v2/resources` and
`assets/magenta-rt-v2/models` as symlinks to the Magenta RealTime 2 assets that
already exist on this machine (by default `~/Documents/Magenta/magenta-rt-v2`).

Nothing here is downloaded or duplicated. The symlinks are intentionally ignored
by git; the script is the reproducible way to recreate them.
