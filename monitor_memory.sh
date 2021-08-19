if [ $# -lt 1 ]; then
    echo usage: ./monitor_memory.sh PID
    exit
fi

pid=$1
interval=1

ps aux | head -1
while true; do
    mem=$(ps aux | grep -E "kamhoua +$pid " | grep -v grep)
    if [ "x$mem" == "x" ]; then break; fi
    echo $mem
    sleep $interval
done

