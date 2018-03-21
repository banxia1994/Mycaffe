#!/usr/bin/env sh

set -e

EXAMPLE=.build_release/examples/siamese
DATA=/data/Fingervein

echo "creating leveldb..."

rm -rf /data/Fingervein/siamese/siamese_train_leveldb
rm -rf /data/Fingervein/siamese/siamese_test_leveldb

$EXAMPLE/conver_Imagener_siamese_data.bin \
       --shuffle \
       $DATA/MMCBNU-6000/ROI_rectangle/ \
      $DATA/train.txt \
      $DATA/trainpair.txt \
       /data/Fingervein/siamese/siamese_train_leveldb
$EXAMPLE/conver_Imagener_siamese_data.bin \
        --shuffle \
        $DATA/MMCBNU-6000/ROI_rectangle/ \
        $DATA/test.txt \
        $DATA/testpair.txt \
        /data/Fingervein/siamese/siamese_test_leveldb
echo "done!"

