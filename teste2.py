import cv2
import numpy as np
import time

webcam = cv2.VideoCapture("videotest.mp4")
webcam.set(cv2.CAP_PROP_POS_MSEC, 5000)                                                    #captura

detector_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # detectar face
detector_body = cv2.CascadeClassifier('fullbody.xml') #detector corpo

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()                         # reconhecer
# carregar o modelo treinado
lbph_classifier.read('lbph_classifier_1classe_leonardo.yml')

#definindo dimensões das imagens para ficar igual ao do treinamento
LARGURA_TREINO = 320 
ALTURA_TREINO = 243

LIMITE_CONFIANCA = 50           # quanto menor a distancia, melhor

ESTADO = "deteccao"

while True:
    ok, frame = webcam.read()
    if not ok: break

    frame = cv2.resize(frame, (LARGURA_TREINO, ALTURA_TREINO))
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
    ESTADO = "reconhecimento"

    #cont = 0
    #cv2.putText(frame, "Nenhuma Face Encontrada", (25, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
           

    if ESTADO == "nao_reconhecido":
        cont= cont+1
        cv2.putText(frame, 'Nao Identificado', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1)
        ESTADO ="rastreamento"

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

    if ESTADO == "detecta":
        #detectando face
        deteccoes_face = detector_faces.detectMultiScale(img_gray, scaleFactor=1.09, minNeighbors=7)
        x, y, w, h = 0, 0, 0, 0
        if len(deteccoes_face):
            ESTADO = "reconhecimento"
            for x, y, w, h in deteccoes_face:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 200), 1)          # desenha o retângulo
        else:
            ESTADO = "detecta_body"

#-----------------------------------------------------------------------------------------------------

    if ESTADO == "reconhecimento":
        # prevendo
        previsao, distancia = lbph_classifier.predict(img_gray)

        texto_debug = f"Dist:{round(distancia)}"  # texto para imprimir distancia na imagem
        cv2.putText(frame, texto_debug, (30, ALTURA_TREINO -60), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 75, 0))

        # Decisão
        if  distancia < LIMITE_CONFIANCA:
                ESTADO = "identificando"
                inicio_identificado = time.time()
                cv2.putText(frame, 'Leonardo', (25, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
                
        else:
                ESTADO ="nao_identificando"
                inicio_nao_identificado = time.time()
                cv2.putText(frame, 'Não identificdo', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1)
#-----------------------------------------------------------------------------------------------------

    if ESTADO == "detecta_body":
        deteccoes_body = detector_body.detectMultiScale(frame, scaleFactor=1.04, minNeighbors=5)

        if len(deteccoes_body) > 0:
            ESTADO = "rastreamento"
        else:
            ESTADO = "detecta"

#-----------------------------------------------------------------------------------------------------

    if ESTADO == "rastreamento":
        pass

#-----------------------------------------------------------------------------------------------------

    if ESTADO == "identificando":
        # prevendo
        previsao, distancia = lbph_classifier.predict(img_gray)
        
        texto_debug = f"Dist:{round(distancia)}"  # texto para imprimir distancia na imagem
        cv2.putText(frame, texto_debug, (30, ALTURA_TREINO -60), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 75, 0))
        
        if distancia < LIMITE_CONFIANCA:
            identificado_atual = time.time()
            tempo_identificado = inicio_identificado - identificado_atual
            if tempo_identificado > 6.0:
                ESTADO = "imprime_identidade"
                cv2.putText(frame, 'IDENTIFICADO', (25, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
            else:
                ESTADO = "identificando"
                cv2.putText(frame, 'Leonardo', (25, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
        else:
            ESTADO = "nao_identificando"
            cv2.putText(frame, 'Nao Identificado', (25, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
            inicio_nao_identificado = time.time()   # reinicia o tempo para saber se é realmente não identificado 
#-----------------------------------------------------------------------------------------------------

    if ESTADO == "nao_identificando":
        # prevendo
        previsao, distancia = lbph_classifier.predict(img_gray)
        
        texto_debug = f"Dist:{round(distancia)}"  # texto para imprimir distancia na imagem
        cv2.putText(frame, texto_debug, (30, ALTURA_TREINO -60), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 75, 0))
        
        if distancia < LIMITE_CONFIANCA: #reconheceu 
            ESTADO = "identificando"
            inicio_identificado = time.time()   #reinicia a contagem do tempo para saber se é identificado
            cv2.putText(frame, 'Leonardo', (25, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
       
        else: # não reconheceu
            nao_identificado_atual = time.time()
            tempo_nao_identificado = inicio_nao_identificado - nao_identificado_atual

            if tempo_nao_identificado > 10.0: # não reconheceu por tempo suficiente
                ESTADO = "detecta_p_rastrear"
                cv2.putText(frame, 'INICIANDO RASTREAMENTO', (30, ALTURA_TREINO/2 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),1)
            else:   #não reconheceu mas ainda não é tempo suficiente
                ESTADO = "nao_identificando"
                cv2.putText(frame, 'Nao Idenificado', (25, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1)

#-----------------------------------------------------------------------------------------------------

    if ESTADO == "imprime_identidade":
        cv2.putText(frame, 'IDENTIFICADO', ( 30, ALTURA_TREINO/2 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)
        cv2.putText(frame, 'ACESSO PERMITIDO!', ( 20, ALTURA_TREINO/2 +30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),1)

#-----------------------------------------------------------------------------------------------------

    if ESTADO == "detecta_p_rastrear":
        cv2.putText(frame, 'INICIANDO RASTREAMENTO', (30, ALTURA_TREINO/2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),1)
        #detectando face
        deteccoes_face = detector_faces.detectMultiScale(img_gray, scaleFactor=1.09, minNeighbors=7)
        deteccoes_body = detector_body.detectMultiScale(frame, scaleFactor=1.04, minNeighbors=5)

        x, y, w, h = 0, 0, 0, 0
        if len(deteccoes_face) > 0  | len(deteccoes_body) > 0:
            ESTADO = "rastreamento"
            for x, y, w, h in deteccoes_face:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 200), 1)          # desenha o retângulo

            for x_bd, y_bd, w_bd, h_bd in deteccoes_body:
                cv2.rectangle(frame, (x_bd, y_bd), (x_bd + w_bd, y_bd + h_bd), (0, 200, 200), 1)   
        
        else:
            ESTADO = "detecta_p_rastrear"
#------------------------------------------------------------------------------------
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

webcam.release()
cv2.destroyAllWindows()

#------------------------rastreador----------------------------------------------

#cap = cv2.VideoCapture("videotest.mp4")

#ok, frame = webcam.read()
#frame = cv2.resize(frame, (LARGURA_TREINO, ALTURA_TREINO))

detector_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # detectar face
detector_body = cv2.CascadeClassifier('fullbody.xml') #detector corpo


deteccoes = detector_faces.detectMultiScale(frame, scaleFactor=1.09, minNeighbors=5)

while len(deteccoes) == 0:
    ret, frame = webcam.read()
    deteccoes = detector_faces.detectMultiScale(frame, scaleFactor=1.09, minNeighbors=5)

deteccoes_body = detector_body.detectMultiScale(frame, scaleFactor=1.04, minNeighbors=5)

# cria o rastreador CSRT
tracker = cv2.TrackerCSRT_create()
tracker_body = cv2.TrackerCSRT_create()

# inicializa o rastreador
tracker.init(frame, deteccoes[0])

if len(deteccoes_body):
    tracker_body.init(frame, deteccoes_body[0])

cont =0
total =26

while True:
    ok, frame = webcam.read()
    if not ok:
        break
    frame = cv2.resize(frame, (LARGURA_TREINO, ALTURA_TREINO))

    if cont < total:
        success, bbox = tracker.update(frame)
        if len(deteccoes_body):
            success_bd, bbox_bd = tracker_body.update(frame)
            print("Iniciando body")
        else:
            success_bd = False
            print("Nenhum corpo identificado\n")

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (150,0,255), 1)
            cv2.putText(frame, 'Nao Identificado', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1)
            cv2.putText(frame, 'Rastreando', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

        if success_bd:
            x, y, w, h = [int(v) for v in bbox_bd]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,165,255), 2)

        cont = cont+1
    else:   #------------ reinicia o rastreador -------------------------------
        frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        deteccoes = detector_faces.detectMultiScale(frame_cinza, scaleFactor=1.09, minNeighbors=5)
        if len(deteccoes) > 0:
            bbox = tuple(deteccoes[0])
            tracker = cv2.legacy.TrackerCSRT_create()  # recria o rastreador
            tracker.init(frame, bbox)
            #print("Reiniciado com nova bbox:", bbox)
        
        deteccoes_body = detector_body.detectMultiScale(frame_cinza, scaleFactor=1.02, minNeighbors=3)
        if len(deteccoes_body) > 0:
            bbox_bd = tuple(deteccoes_body[0])
            tracker_body = cv2.legacy.TrackerCSRT_create()  # recria o rastreador
            tracker_body.init(frame, bbox_bd)
            print("Reiniciado com nova bbox_bd:", bbox_bd)


        cont = 0

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

webcam.release()
cv2.destroyAllWindows()