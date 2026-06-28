import cv2
import numpy as np
import time

webcam = cv2.VideoCapture("test1.mp4")
#webcam.set(cv2.CAP_PROP_POS_MSEC, 5000)                                                    #captura
#webcam = cv2.VideoCapture("http://192.168.1.4:8080/video")

detector_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # detectar face
detector_body = cv2.CascadeClassifier('fullbody.xml') #detector corpo

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()                         # reconhecer
# carregar o modelo treinado
lbph_classifier.read('lbph_classifier_1classe_leonardo.yml')

#definindo dimensões das imagens para ficar igual ao do treinamento
LARGURA_TREINO = 320 
ALTURA_TREINO = 243

tracker_face = cv2.TrackerCSRT_create()
tracker_body = cv2.TrackerCSRT_create()

rastreando_face = False
rastreando_body = False

contador_reidentificacao = 0
FRAMES_REIDENTIFICAR = 12

LIMITE_CONFIANCA = 60           # quanto menor a distancia, melhor

inicio_perda_identificacao = None
inicio_identificado = None

ESTADO = "detecta"

while True:
    ok, frame = webcam.read()
    if not ok: break

    frame = cv2.resize(frame, (LARGURA_TREINO, ALTURA_TREINO))
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                

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

    elif ESTADO == "reconhecimento":
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

    elif ESTADO == "detecta_body":
        deteccoes_body = detector_body.detectMultiScale(img_gray, scaleFactor=1.04, minNeighbors=5)

        if len(deteccoes_body) > 0:
            ESTADO = "rastreamento"
        else:
            ESTADO = "detecta"

#-----------------------------------------------------------------------------------------------------

    elif ESTADO == "rastreamento":

        cv2.putText(frame,
                    "RASTREANDO",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,255),
                    2)

        # -------------------------------------------------
        # Atualiza tracker
        # -------------------------------------------------

        if rastreando_face:
            success, bbox = tracker_face.update(frame)
        else:
            success, bbox = tracker_body.update(frame)

        if not success:

            rastreando_face = False
            rastreando_body = False

            ESTADO = "detecta_p_rastrear"

        else:

            x, y, w, h = [int(v) for v in bbox]

            cv2.rectangle(frame,
                        (x,y),
                        (x+w,y+h),
                        (255,0,255),
                        2)

            contador_reidentificacao += 1

            if contador_reidentificacao >= FRAMES_REIDENTIFICAR:

                contador_reidentificacao = 0

                #===========================================================
                # CASO 1 - O tracker está seguindo uma FACE
                #===========================================================

                if rastreando_face:

                    faces = detector_faces.detectMultiScale(
                        img_gray,
                        scaleFactor=1.09,
                        minNeighbors=7)

                    if len(faces):

                        print("Atualizando tracker pela face")

                        xf,yf,wf,hf = faces[0]

                        tracker_face = cv2.TrackerCSRT_create()
                        tracker_face.init(frame,(xf,yf,wf,hf))

                        # reconhecimento

                        previsao, distancia = lbph_classifier.predict(img_gray)

                        if distancia < LIMITE_CONFIANCA:
                            print("Face reconhecida")
                            if inicio_identificado is None:
                                inicio_identificado = time.time()

                            if time.time() - inicio_identificado >= 3.5:

                                ESTADO = "imprime_identidade"

                        else:
                            print("Face não reconhecida")
                            inicio_identificado = None

                    else:

                        print("Nenhuma face encontrada.")
                        print("Continuando com tracker atual.\n")

                #===========================================================
                # CASO 2 - O tracker está seguindo um CORPO
                #===========================================================

                else:

                    encontrou_face = False

                    # ---------------------------------------
                    # procura face DENTRO do corpo
                    # ---------------------------------------

                    roi = img_gray[
                        max(0,y):min(y+h,img_gray.shape[0]),
                        max(0,x):min(x+w,img_gray.shape[1])
                    ]

                    if roi.size > 0:

                        faces = detector_faces.detectMultiScale(
                            roi,
                            scaleFactor=1.09,
                            minNeighbors=7)

                        if len(faces):

                            print("Face encontrada dentro da ROI do corpo.")

                            xf,yf,wf,hf = faces[0]

                            xf += x
                            yf += y

                            encontrou_face = True

                    # ---------------------------------------
                    # Se não encontrou na ROI,
                    # procura na imagem inteira
                    # ---------------------------------------

                    if not encontrou_face:

                        print("Face não encontrada na ROI.")
                        print("Tentando imagem inteira...")

                        faces = detector_faces.detectMultiScale(
                            img_gray,
                            scaleFactor=1.09,
                            minNeighbors=7)

                        if len(faces):

                            xf,yf,wf,hf = faces[0]

                            encontrou_face = True

                    # ---------------------------------------
                    # Encontrou uma face
                    # ---------------------------------------

                    if encontrou_face:

                        print("Atualizando tracker para FACE.")

                        tracker_face = cv2.TrackerCSRT_create()

                        tracker_face.init(frame,
                                        (xf,yf,wf,hf))

                        rastreando_face = True
                        rastreando_body = False

                        previsao, distancia = lbph_classifier.predict(img_gray)

                        if distancia < LIMITE_CONFIANCA:
                            print("Face reconhecida")
                            if inicio_identificado is None:
                                inicio_identificado = time.time()

                            if time.time() - inicio_identificado >= 3.5:

                                ESTADO = "imprime_identidade"

                        else:
                            print("Face não reconhecida")
                            inicio_identificado = None

                    else:

                        # ---------------------------------------
                        # Ainda sem face.
                        # Atualiza o tracker usando um novo corpo.
                        # ---------------------------------------

                        corpos = detector_body.detectMultiScale(
                            img_gray,
                            scaleFactor=1.04,
                            minNeighbors=5)

                        if len(corpos):

                            print("Atualizando tracker pelo corpo.")

                            xb,yb,wb,hb = corpos[0]

                            tracker_body = cv2.TrackerCSRT_create()

                            tracker_body.init(frame,
                                            (xb,yb,wb,hb))

                            rastreando_body = True
                            rastreando_face = False

                        else:

                            print("Nenhum corpo encontrado.")
                            print("Continuando tracker atual.\n")

#-----------------------------------------------------------------------------------------------------

    elif ESTADO == "identificando":
        # prevendo
        previsao, distancia = lbph_classifier.predict(img_gray)
        
        texto_debug = f"Dist:{round(distancia)}"  # texto para imprimir distancia na imagem
        cv2.putText(frame, texto_debug, (30, ALTURA_TREINO -60), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 75, 0))
        
        if distancia < LIMITE_CONFIANCA:
            identificado_atual = time.time()
            tempo_identificado = identificado_atual - inicio_identificado 
            if tempo_identificado > 3.5:
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

    elif ESTADO == "nao_identificando":
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
            tempo_nao_identificado = nao_identificado_atual - inicio_nao_identificado

            if tempo_nao_identificado > 4.5: # não reconheceu por tempo suficiente
                ESTADO = "detecta_p_rastrear"
                cv2.putText(frame, 'INICIANDO RASTREAMENTO', (30, ALTURA_TREINO // 2 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),1)
            else:   #não reconheceu mas ainda não é tempo suficiente
                ESTADO = "nao_identificando"
                cv2.putText(frame, 'Nao Idenificado', (25, 40 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1)

#-----------------------------------------------------------------------------------------------------

    elif ESTADO == "imprime_identidade":
        previsao, distancia = lbph_classifier.predict(img_gray)

        texto_debug = f"Dist:{round(distancia)}"
        cv2.putText(frame, texto_debug, (30, ALTURA_TREINO - 60), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 75, 0))

        if distancia < LIMITE_CONFIANCA:
            # reconhecimento continua valido, reinicia o temporizador de perda
            inicio_perda_identificacao = None

            cv2.putText(frame, "IDENTIFICADO", (30, ALTURA_TREINO//2),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2)
            cv2.putText(frame, "ACESSO PERMITIDO", (20, ALTURA_TREINO//2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:
            # primeira falha de reconhecimento
            if inicio_perda_identificacao is None:
                inicio_perda_identificacao = time.time()

            tempo_perdido = time.time() - inicio_perda_identificacao

            if tempo_perdido > 3.0:
                inicio_nao_identificado = time.time()
                inicio_perda_identificacao = None
                ESTADO = "nao_identificando"
            else:
                # ainda dentro da tolerância permanece em imprime_identidade
                cv2.putText(frame, "IDENTIFICADO", (30, ALTURA_TREINO//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, "ACESSO PERMITIDO", (20, ALTURA_TREINO//2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#-----------------------------------------------------------------------------------------------------

    elif ESTADO == "detecta_p_rastrear":

        cv2.putText(frame, "INICIANDO RASTREAMENTO", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        deteccoes_face = detector_faces.detectMultiScale(img_gray, scaleFactor=1.02, minNeighbors=9)

        if len(deteccoes_face):
            tracker_face = cv2.TrackerCSRT_create()
            tracker_face.init(frame, tuple(deteccoes_face[0]))

            rastreando_face = True
            rastreando_body = False

            contador_reidentificacao = 0

            ESTADO = "rastreamento"

        else:
            deteccoes_body = detector_body.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=9)

            if len(deteccoes_body):
                tracker_body = cv2.TrackerCSRT_create()
                tracker_body.init(frame, tuple(deteccoes_body[0]))

                rastreando_body = True
                rastreando_face = False

                contador_reidentificacao = 0

                ESTADO = "rastreamento"
#------------------------------------------------------------------------------------
    cv2.imshow('Video', frame)

    key = cv2.waitKey(70)
    if key == 27: # ESC
        break

webcam.release()
cv2.destroyAllWindows()
