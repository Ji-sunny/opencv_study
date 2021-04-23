import cv2
import numpy as np

win_name = 'scan'
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)


def mouseHandler(event, x, y, flags, param):
    global pts_cnt
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
        cv2.imshow(win_name, img)
        pts[pts_cnt] = [x, y]
        pts_cnt += 1
        if pts_cnt == 4:
            sm = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            topleft = pts[np.argmin(sm)]
            bottom = pts[np.argmax(sm)]
            topright = pts[np.argmin(diff)]
            bottomLeft = pts[np.argmax(diff)]

            pts1 = np.float32([topleft, topright, bottom, bottomLeft])

            w1 = abs(bottom[0] - bottomLeft[0])
            w2 = abs(topright[0] - topleft[0])
            h1 = abs(topright[1] - bottom[1])
            h2 = abs(topleft[1] - bottomLeft[1])

            width = max([w1, w2])
            height = max([h1, h2])

            pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img, mtrx, (width, height))
            cv2.imshow('scanning', result)


cap = cv2.VideoCapture(0)
captured = False

if cap.isOpened():
    delay = int(1000 / cap.get(cv2.CAP_PROP_FPS))
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow("Capture", img)
            key = cv2.waitKey(delay)
            if key & 0xFF == 27:
                print(key)
                break
            elif key == ord('c'):
                captured = True
                break
        else:
            break
else:
    print("비디오 안열림")

if captured:
    cap.release()
    while True:
        cv2.imshow('Capture', img)
        cv2.setMouseCallback('Capture', mouseHandler)
        key = cv2.waitKey(delay)
        if key & 0xFF == 27:
            print(key)
            break

cap.release()
cv2.destroyAllWindows()