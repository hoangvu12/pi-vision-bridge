#!/bin/bash
set -e
cd "$(dirname "$0")"

# drawtext needs an explicit fontfile on this Windows ffmpeg build
# (fontconfig has no default config); the \: escape keeps the drive-letter
# colon from splitting the filter options.

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc=size=640x360:rate=10:duration=4" \
  -vf "drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='SCENE 1 RED':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=red@0.8" \
  -c:v libx264 -pix_fmt yuv420p part1.mp4

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc2=size=640x360:rate=10:duration=4" \
  -vf "drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='SCENE 2 GREEN':fontcolor=black:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=green@0.8" \
  -c:v libx264 -pix_fmt yuv420p part2.mp4

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "smptebars=size=640x360:rate=10:duration=4" \
  -vf "drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':text='SCENE 3 BLUE':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=blue@0.8" \
  -c:v libx264 -pix_fmt yuv420p part3.mp4

ffmpeg -hide_banner -loglevel error -y -i part1.mp4 -i part2.mp4 -i part3.mp4 \
  -filter_complex "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]" -map "[v]" -c:v libx264 -pix_fmt yuv420p test-video.mp4

rm -f part1.mp4 part2.mp4 part3.mp4 probe-drawtext.mp4
echo "duration: $(ffprobe -v error -show_entries format=duration -of csv=p=0 test-video.mp4)s"
ls -la test-video.mp4
