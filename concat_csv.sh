awk '(NR == 1) || (FNR > 1)' grid1*/*.csv > out.csv
