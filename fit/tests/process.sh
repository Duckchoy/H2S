#!/bin/bash
grep "fitted value" $1 | awk '{print $3,0.0}' 
echo " "
grep "fitted value" $1 | awk '{print $5,$7}'

