awk '(NR == 1) || (FNR > 1)' *.csv > out.csv
