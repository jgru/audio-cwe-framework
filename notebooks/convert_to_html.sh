#!/bin/sh
CURR_DIR=./
files=($CURR_DIR/*)
 
for ((i=0; i<${#files[@]}; i++));
    do
    ipython nbconvert --to html "${files[$i]}"
done