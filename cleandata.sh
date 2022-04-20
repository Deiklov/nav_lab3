#!/bin/sh
touch Rover1.cleanest && cat ROVER1.GK | sed -r 's/ +/,/g' | sed s/,// > Rover1.cleanest