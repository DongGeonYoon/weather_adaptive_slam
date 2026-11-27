#!/usr/bin/env bash
# ~1.5 min presentation highlight from render_presentation.py outputs.
# Driving part of each clip at SPEED x, the trailing 4 s metric card at 1x.
# usage: sim/make_highlight.sh <presentation dir> [SPEED]
set -euo pipefail
P="$(cd "$1" && pwd)"; SPEED="${2:-3}"
CLIPS=(compare_valD_all_weather compare_valC_rain compare_valD_snow)
TMP="$P/_hl"; mkdir -p "$TMP"
ENC=(-c:v libx264 -pix_fmt yuv420p -crf 20 -r 30 -an)

ffmpeg -y -loglevel error -i "$P/intro.mp4" -t 4 "${ENC[@]}" "$TMP/00.mp4"
i=1
for c in "${CLIPS[@]}"; do
    d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$P/$c.mp4")
    m=$(python3 -c "print(max(0.0, $d - 4.0))")
    ffmpeg -y -loglevel error -i "$P/$c.mp4" -filter_complex \
        "[0:v]trim=0:$m,setpts=(PTS-STARTPTS)/$SPEED[a];[0:v]trim=start=$m,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[v]" \
        -map "[v]" "${ENC[@]}" "$TMP/$(printf %02d $i).mp4"
    i=$((i + 1))
done
ffmpeg -y -loglevel error -i "$P/summary.mp4" -t 8 "${ENC[@]}" "$TMP/99.mp4"

ls "$TMP"/*.mp4 | sort | sed "s|^|file '|; s|$|'|" > "$TMP/list.txt"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$TMP/list.txt" -c copy "$P/highlight.mp4"
rm -rf "$TMP"
echo "wrote $P/highlight.mp4 ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$P/highlight.mp4") s)"
