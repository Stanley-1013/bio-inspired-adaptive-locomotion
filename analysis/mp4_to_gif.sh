#!/usr/bin/env bash
# Convert recorded mp4 clips to small, GitHub-inline-friendly GIFs.
# Uses the ffmpeg bundled with imageio-ffmpeg (no system ffmpeg needed).
# Full-quality mp4s are kept out of git (see results/**/videos/.gitignore) and
# linked externally; the GIFs are committed for inline README playback.
#
# Usage: bash analysis/mp4_to_gif.sh <in.mp4> <out.gif> [width] [fps] [start] [dur]
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate sata
FFMPEG="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
IN="$1"; OUT="$2"; W="${3:-480}"; FPS="${4:-12}"; START="${5:-0}"; DUR="${6:-8}"
PAL="$(mktemp --suffix=.png)"
# two-pass palette for clean small GIFs
"$FFMPEG" -y -ss "$START" -t "$DUR" -i "$IN" \
  -vf "fps=$FPS,scale=$W:-1:flags=lanczos,palettegen" "$PAL" 2>/dev/null
"$FFMPEG" -y -ss "$START" -t "$DUR" -i "$IN" -i "$PAL" \
  -lavfi "fps=$FPS,scale=$W:-1:flags=lanczos[x];[x][1:v]paletteuse" "$OUT" 2>/dev/null
rm -f "$PAL"
echo "  $(du -h "$OUT" | cut -f1)  $OUT"
