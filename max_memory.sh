if [ $# -lt 1 ]; then
    echo usage: ./max_memory.sh log_file
    exit
fi

log_file=$1

#awk '{print $6}' $log_file
awk -v max=0 'NR>1 {if($6>max){max=$6}}END{print max}'  $log_file


#awk 'NR>1' $log_file | awk -v max=0 '{if($6>max){max=$6}}END{print max}'  $log_file

#sort -nrk1,1 filename | head -1 | cut -d ' ' -f3
