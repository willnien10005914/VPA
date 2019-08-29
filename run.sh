cleanup()
{
  echo "Caught Signal ... cleaning up."
  kill -9 `ps aux | grep 'rtsp.py' | grep -v grep | awk '{print $2}'`
}

trap 'cleanup' INT
#pip install -r requirements.txt
python my_rtsp.py --input 'rtsp://192.168.1.125/unicast/video:1' --output a.mp4
