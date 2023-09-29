#!/bin/bash

cd $SLURM_SUBMIT_DIR

# ./bitpolymul-test 15
# ./bitpolymul-test 16
# ./bitpolymul-test 17
# ./bitpolymul-test 18
# ./bitpolymul-test 19
./bitpolymul-test 20
./bitpolymul-test 21
./bitpolymul-test 22
./bitpolymul-test 23

# compute-sanitizer --tool memcheck ./bitpolymul-test 23 &> out
# nsys profile --stats=true ./bitpolymul-test 23 &> prof