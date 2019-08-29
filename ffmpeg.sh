ffserver -f myrtsp.conf &
ffmpeg -re -i /dev/video0 -c copy http://127.0.0.1:8090/feed1.ffm
#ffplay "rtsp://127.0.0.1:8554/test.mpeg4"


#ffmpeg -i /dev/video0 -f mpegts udp://127.0.0.1:8554

#ffplay "udp://127.0.0.1:8554"