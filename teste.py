import cv2
import numpy as np

webcam = cv2.VideoCapture(0)                                                   #captura
detector_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # detectar face
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()                         # reconhecer
rastreador = cv2.legacy.TrackerKCF_create()                                  # rastrear

# carregar o modelo treinado
lbph_classifier.read('lbph_classifier_1classe_leonardo.yml')
#definindo dimensões das imagens para ficar igual ao do treinamento
LARGURA_TREINO = 320 
ALTURA_TREINO = 243

LIMITE_CONFIANCA = 50           # quanto menor a distancia, melhor

while True:
    ok, frame = webcam.read()
    if not ok: break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectando face
    deteccoes = detector_faces.detectMultiScale(img_gray, scaleFactor=1.09, minNeighbors=7)

    if len(deteccoes):
        x,y,w,h = 0,0,0,0
        for x, y, w, h in deteccoes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 200), 1)          # desenha o retângulo

        # redimensinando para o tamanho que foi treinado
        face_resized = cv2.resize(img_gray, (LARGURA_TREINO, ALTURA_TREINO))
                
        # o predict retorna (label, distancia)
        previsao, distancia = lbph_classifier.predict(face_resized)

        texto_debug = f"Dist:{round(distancia)}"  # texto para imprimir distancia na imagem
        
        cv2.putText(frame, texto_debug, (x, y+h+15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        # Decisão
        if  distancia < LIMITE_CONFIANCA:
            cv2.putText(frame, 'Leonardo', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
                    
        else:
            cv2.putText(frame, 'Não Identificado', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1)

    else:
        cv2.putText(frame, "Nenhuma Face Encontrada", (25, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))


    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

webcam.release()
cv2.destroyAllWindows()