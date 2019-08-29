# -*- coding: utf-8 -*-
import cv2
import time
import threading


URL1 = "rtsp://192.168.1.125/unicast/video:1"
URL = "rtsp://127.0.0.1:8554/test.mpeg4"
#URL = "/dev/video0"


# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False

        # 攝影機連接。
        self.capture = cv2.VideoCapture(URL)

    def start(self):
        # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        # 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')

    def getframe(self):
        # 當有需要影像時，再回傳最新的影像。
        return self.Frame

    def getStatus(self):
        return self.status

    def queryframe(self):
        while (not self.isstop):
            self.status, tmp = self.capture.read()
            if (self.status == True) :
                self.Frame = tmp

        self.capture.release()



def open_webcam_1():
    print("open_webcam_1")
    # 連接攝影機
    ipcam = ipcamCapture(URL1)

    # 啟動子執行緒
    ipcam.start()

    # 暫停1秒，確保影像已經填充
    time.sleep(1)

    # 使用無窮迴圈擷取影像，直到按下Esc鍵結束
    try:
        while True:


            # 使用 getframe 取得最新的影像
            I = ipcam.getframe()

            # gray
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Image', I)


            if cv2.waitKey(10) == 27:
                cv2.destroyAllWindows()
                ipcam.stop()
                break
            '''
            if cv2.waitKey(1000) == ord('p'):
                print("snapshot!!!!!!!!!!!!!!!")
                threading.Thread(target=save_picture_1, daemon=True, args=(I,)).start()
            '''
    except:
        print("An exception occurred")
        open_webcam_1()


def open_webcam_2():
    print("open_webcam_2")
    cap = cv2.VideoCapture(URL)
    ret, frame = cap.read()
    try:
        while ret:
            ret, frame = cap.read()
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        print("An exception occurred")
        open_webcam_2()


    cv2.destroyAllWindows()
    cap.release()


def open_picture_1():
    print("open_picture_1")
    image = cv2.imread("/home/bu10/Pictures/cat.jpeg")
    cv2.imshow("pic1", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def open_picture_2():
    print("open_picture_2")
    image = cv2.imread("/home/bu10/Pictures/car.jpg")
    cv2.imshow("pic2", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def save_video_1():
    print("save_video_1")
    cap = cv2.VideoCapture(URL1)

    ret, frame = cap.read()

    f = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('a.mp4', f, 16.0, (1280, 720))

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            continue
        #cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            out.release()
            break

def save_picture_1(frame):
    print("save_picture_1")
    cv2.imwrite('a.jpg', frame)

def test():
    while True:
        time.sleep(3)
        print("test")

def main():
    #threading.Thread(target=save_video_1, daemon=True, args=()).start()
    threading.Thread(target=open_webcam_1, daemon=True, args=()).start()
    #threading.Thread(target=open_webcam_2, daemon=True, args=()).start()

    #threading.Thread(target=open_picture_1, daemon=True, args=()).start()
    #threading.Thread(target=open_picture_2, daemon=True, args=()).start()
    test()
    #open_picture_1()



if __name__ == '__main__':
    main()



__author__ = "Will Nien"
__email__ = "will.nien@quantatw.com"
__version__ = "1.0.0"