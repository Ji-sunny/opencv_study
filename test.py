import pickle
import numpy as np
with open("./model/number.model", "rb") as f:
    mlp2 = pickle.load(f)

import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bin_img = cv2.threshold(g_img, 110, 255, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(bin_img,
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # CHAIN_APPROX_SIMPLE 테두리를 잡아줌
            try:
                for i in range(len(contours)):
                    contour = contours[i]
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if radius > 3:
                        xs, xe = int(x - radius), int(x + radius)
                        ys, ye = int(y - radius), int(y + radius)
                        cv2.rectangle(bin_img, (xs, ys), (xe, ye), (200, 0, 0), 1)
                        roi = bin_img[ys:ye, xs:xe]
                        dst = cv2.resize(roi, dsize=(50, 50),
                                         interpolation=cv2.INTER_AREA)
                        dst = cv2.resize(dst, dsize=(16, 16),
                                         interpolation=cv2.INTER_AREA)
                        A = np.zeros((20, 20))
                        A[2:-2, 2:-2] = dst[:, :]
                        A = A.reshape(-1, 400)
                        num = mlp2.predict(A)
                        cv2.putText(bin_img, str(num), (xs, ys), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 0))
            except Exception as e:
                print(e)
            cv2.imshow("Image", bin_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            print("Np Frame")
            break
else:
    print("Camera not opened")

cap.release()
cv2.destroyAllWindows()
