import cv2
import time

webcam = cv2.VideoCapture(0)
cont = 0
start = time.time()

while True:
    
    ok, frame = webcam.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    end = time.time()
    if end-start > 1:
        start = time.time()
        cv2.imwrite(f"leoTeste/face{cont}.jpeg",img_gray)
        print(f"captura {cont}")
        cont = cont + 1
    
    cv2.imshow('Video', frame)

    key = cv2.waitKey(5)
    if key == 27:
        print("Tecla ESC precionada. Encerrando.")
        break

webcam.release()
cv2.destroyAllWindows()