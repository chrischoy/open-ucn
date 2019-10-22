#!/usr/bin/bash
# Usage : bash download_yfcc.sh [path/to/download/dataset]
# Example : baseh download_yfcc.sh /root/data

DATA_DIR=$1

DATA_NAME=oanet_data
FILE_NAME=raw_data
OUTPUT_NAME=raw_data_yfcc.tar.gz

cd $DATA_DIR

if [ ! -d download_data_$DATA_NAME ]; then
    mkdir -p download_data_$DATA_NAME
fi

let CHUNK_START=0
let CHUNK_END=8


for ((i=CHUNK_START;i<=CHUNK_END;i++)); do
    IDX=$(printf "%03d" $i)
    URL=research.altizure.com/data/$DATA_NAME/$FILE_NAME.tar.$IDX
    wget -c $URL -P download_data_$DATA_NAME
    echo $URL
done


cat download_data_oanet_data/*.tar.* > $OUTPUT_NAME
rm -r download_data_oanet_data

# Unzip
tar -xvzf $OUTPUT_NAME
mv raw_data/yfcc100m .
rm -rf raw_data

cd -