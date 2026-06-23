import cv2
import numpy as np

webcam = cv2.VideoCapture(0)
lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
detector_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Carrega o modelo treinado
lbph_classifier.read('lbph_classifier_1classe_leonardo.yml')

# --- CONFIGURAÇÕES ---
# IMPORTANTE: Este tamanho DEVE ser EXATAMENTE o mesmo usado no treinamento no Colab.
# Se no Colab você usou 200x200, mude aqui para 200, 200.
LARGURA_TREINO = 320 
ALTURA_TREINO = 243

# Limiar de confiança. No LBPH, quanto MENOR, melhor.
# Geralmente, abaixo de 50 é uma correspondência perfeita.
# Acima de 80 ou 90 geralmente é desconhecido ou falso positivo.
LIMITE_CONFIANCA = 66

while True:
    ok, frame = webcam.read()
    if not ok: break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    deteccoes = detector_faces.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    if len(deteccoes) > 0:
        for x, y, w, h in deteccoes:
            # Extrai a face
            face_roi = img_gray[y:y+h, x:x+w]
            
            # Redimensiona para o padrão do treinamento
            face_resized = cv2.resize(face_roi, (LARGURA_TREINO, ALTURA_TREINO))
            
            # O predict retorna (label, distancia)
            previsao, distancia = lbph_classifier.predict(face_resized)

            # Formata o texto para vermos a "confiança" na tela
            texto_debug = f"ID:{previsao} Dist:{round(distancia)}"
            
            # Desenha o retângulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, texto_debug, (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

            # LÓGICA DE DECISÃO
            # Verificamos se é a classe 0 E se a distância é aceitável
            if previsao == 0 and distancia < LIMITE_CONFIANCA:
                cv2.putText(frame, 'Leonardo', (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            
            # Se for classe 1 e a distância for aceitável
            elif previsao == 1 and distancia < LIMITE_CONFIANCA:
                 cv2.putText(frame, 'Outra Pessoa', (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            
            # Se a distância for muito alta, o modelo está "chutando"
            else:
                cv2.putText(frame, 'Desconhecido', (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

    else:
        cv2.putText(frame, 'Sem rosto', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv2.imshow('Video', frame)

    key = cv2.waitKey(5)
    if key == 27: # ESC
        break

webcam.release()
cv2.destroyAllWindows()