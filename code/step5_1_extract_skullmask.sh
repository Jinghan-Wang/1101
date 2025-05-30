#!/bin/bash
#heran yang

for file in `ls`

do

  echo "ExtractSkull $file ! \n"


  INPUT=/data/ZZJ/code/1101/dataset/process/Training/MRI/$file
  OUTPUT=/data/ZZJ/code/1101/dataset/process/fsl_result/$file

#  bet2 ${INPUT} ${OUTPUT} -o -m -f 0.5
  bet2 "${INPUT}" "${OUTPUT}" -o -m -f 0.5

done

