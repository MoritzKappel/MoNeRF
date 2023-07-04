#! /usr/bin/env bash
cd $(dirname $(realpath "$0"))/../dataset
curl -O https://nextcloud.mpi-klsb.mpg.de/index.php/s/EHtctQJZDWWcfqj/download
unzip download && rm download
find . -name "*.zip" | while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; rm "$filename"; done;