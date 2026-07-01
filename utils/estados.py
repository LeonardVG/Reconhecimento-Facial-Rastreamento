import cv2
import time
from utils import config
import os
from datetime import datetime

class Estados:
 # MÉTODOS AUXILIARES
    def _prever_e_desenhar_debug(self, frame, img_gray):
        """Faz a predição e já desenha a distância na tela. Retorna a distância."""
        previsao, distancia = self.lbph_classifier.predict(img_gray)
        texto_debug = f"Dist:{round(distancia)}"
        cv2.putText(frame, texto_debug, (30, config.ALTURA_TREINO - 60), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 75, 0))
        return distancia

    def _iniciar_rastreamento_face(self, frame, bbox):
        """Centraliza a lógica de iniciar o rastreador de face."""
        self.tracker_face = cv2.TrackerCSRT_create()
        self.tracker_face.init(frame, bbox)
        self.rastreando_face = True
        self.rastreando_body = False
        self.contador_reidentificacao = 0

    def _iniciar_rastreamento_body(self, frame, bbox):
        """Centraliza a lógica de iniciar o rastreador de corpo."""
        self.tracker_body = cv2.TrackerCSRT_create()
        self.tracker_body.init(frame, bbox)
        self.rastreando_body = True
        self.rastreando_face = False
        self.contador_reidentificacao = 0

    def _tentar_reconhecimento(self, img_gray, tempo_limite):
        """Tenta reconhecer a face recém detectada durante o rastreamento."""
        previsao, distancia = self.lbph_classifier.predict(img_gray)
        if distancia < config.LIMITE_CONFIANCA: #se conseguiu reconheer verifica se ficou tempo suficiente reconhecendo para diminuir falsos positivos
            print("Face reconhecida")
            if self.inicio_identificado is None:    # se é o primeiro frame que reconheceu depois de rastrear
                self.inicio_identificado = time.time()
            if time.time() - self.inicio_identificado >= tempo_limite:   #se  ficou tempo suficiente reconhecendo
                self.estado = config.EstadoPipeline.IMPRIME_IDENTIDADE
        else:   # se não conseguiu reconhecer nova deteccao continua rastreando e zera time
            print("Face não reconhecida")
            self.inicio_identificado = None

    def _reavaliar_rastreador_face(self, frame, img_gray):
        """Lógica executada para reavaliar a face após N frames"""
        faces = self.detector_faces.detectMultiScale(img_gray, scaleFactor=1.03, minNeighbors=7)
        if len(faces):  # se conseguiu detectar uma face, reinicia o tracker com a face nova (ou na posição correta)
            print("Atualizando tracker pela face")
            self._iniciar_rastreamento_face(frame, tuple(faces[0]))
            self._tentar_reconhecimento(img_gray, tempo_limite=3.5) # tenta reconhecer essa face nova detectada
        else:   # se nenhuma face foi encontrada continua atualizando o tracker onde esta
            print("Nenhuma face encontrada.\nContinuando com tracker atual.\n")

    def _reavaliar_rastreador_body(self, frame, img_gray, x, y, w, h):
        """Lógica executada para tentar achar rosto no corpo ou reavaliar corpo."""
        encontrou_face = False
        # cria um roi para tentar detectar face dentro do corpo para rastrear preferencialmente pela face
        roi = img_gray[max(0, y):min(y+h, img_gray.shape[0]), max(0, x):min(x+w, img_gray.shape[1])]
        
        if roi.size > 0:
            faces = self.detector_faces.detectMultiScale(roi, scaleFactor=1.09, minNeighbors=7)
            if len(faces):
                print("Face encontrada dentro da ROI do corpo.")
                xf, yf, wf, hf = faces[0]
                face_bbox = (xf + x, yf + y, wf, hf)
                encontrou_face = True

        #se nã conseguiu detectar no roi do crpo, msm assim tenta detectar na imagem inteira pq a detecção do corpo pode estar em lugar falso
        if not encontrou_face:
            print("Face não encontrada na ROI.\nTentando imagem inteira...")
            faces = self.detector_faces.detectMultiScale(img_gray, scaleFactor=1.09, minNeighbors=7)
            if len(faces):
                face_bbox = tuple(faces[0])
                encontrou_face = True

        if encontrou_face:
            print("Atualizando tracker para FACE.")
            self._iniciar_rastreamento_face(frame, face_bbox)
            self._tentar_reconhecimento(img_gray, tempo_limite=3.0)
        else:
            corpos = self.detector_body.detectMultiScale(img_gray, scaleFactor=1.04, minNeighbors=5)
            if len(corpos):
                print("Atualizando tracker pelo corpo.")
                self._iniciar_rastreamento_body(frame, tuple(corpos[0]))
            else:
                print("Nenhum corpo encontrado.\nContinuando tracker atual.\n")

    def _salvar_imagens_rastreamento(self, frame, x, y, w, h):
        """Salva o frame e o crop do objeto rastreado periodicamente."""
        agora = time.time()
        
        # verifica se já passou tempo suficiente desde a ultima foto
        if agora - self.ultimo_salvamento >= config.INTERVALO_SALVAMENTO:
            self.ultimo_salvamento = agora
            
            # garantir que o retângulo do recorte não tente sair da tela para evita crash
            y_inicio = max(0, y)
            y_fim = min(frame.shape[0], y + h)
            x_inicio = max(0, x)
            x_fim = min(frame.shape[1], x + w)
            
            # Recorta a imagem (bbox)
            bbox_img = frame[y_inicio:y_fim, x_inicio:x_fim]
            
            # cria um nome único usando a data e hora atual
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_frame = os.path.join(config.DIR_OUTPUT, f"{timestamp}_frame.jpg")
            nome_bbox = os.path.join(config.DIR_OUTPUT, f"{timestamp}_bbox.jpg")
            
            # salva as imagens
            cv2.imwrite(nome_frame, frame)
            
            # só salva o bbox se ele tiver um tamanho válido par evita erros se a detecção falhar
            if bbox_img.size > 0:
                cv2.imwrite(nome_bbox, bbox_img)
                
            print(f"Captura salva! ({timestamp})")

#=======================================================================================================================
    # MÉTODOS DE ESTADO 

    def estado_detecta(self, frame, img_gray):
        deteccoes_face = self.detector_faces.detectMultiScale(img_gray, scaleFactor=1.02, minNeighbors=7)
        if len(deteccoes_face):     # se detectou um rosto, vai pro estado reconhecimento
            self.estado = config.EstadoPipeline.RECONHECIMENTO
            for x, y, w, h in deteccoes_face:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 200), 1)
        else:                       # se não detectou tenta detectar um corpo
            self.estado = config.EstadoPipeline.DETECTA_BODY

    def estado_reconhecimento(self, frame, img_gray):
        # tenta reconhecer (reaproveitando método auxiliar)
        distancia = self._prever_e_desenhar_debug(frame, img_gray)

        if distancia < config.LIMITE_CONFIANCA:  # se reconheceu vai pra identificando, para ter certeza que n é um falso positivo
            self.estado = config.EstadoPipeline.IDENTIFICANDO   
            self.inicio_identificado = time.time()
            cv2.putText(frame, 'Leonardo', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        else:   # se n reconheceu vai pra nao_identificando para ter certea que é uma pessoa desconhecida
            self.estado = config.EstadoPipeline.NAO_IDENTIFICANDO
            self.inicio_nao_identificado = time.time()
            cv2.putText(frame, 'Não identificado', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    def estado_detecta_body(self, img_gray):    # se não detectou um rosto no inicio, vem tentar detectar um corpo
        deteccoes_body = self.detector_body.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=3)
        if len(deteccoes_body) > 0: #se detectou um corpo -> rastrea
            self.estado = config.EstadoPipeline.RASTREAMENTO
        else:   # n consegui detectar um corpo -> volta a tentar detectar face
            self.estado = config.EstadoPipeline.DETECTA

    def estado_rastreamento(self, frame, img_gray): # se não reconheceu, ou no inicio não consegui detectar face
        cv2.putText(frame, "RASTREANDO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.rastreando_face:
            success, bbox = self.tracker_face.update(frame)
        else:
            success, bbox = self.tracker_body.update(frame)

        # se o tracker não conseguir mais rastrear
        if not success: 
            self.rastreando_face = False
            self.rastreando_body = False
            self.estado = config.EstadoPipeline.DETECTA_P_RASTREAR
        else: #conseguiu rastrear no frame atual
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            self._salvar_imagens_rastreamento(frame, x, y, w, h)
            self.contador_reidentificacao += 1

            #verificar se tem que tentar detectar novamente (nao passo tempo suficiente reconhecendo) para reiniciar o tracker com nova deteccao
            if self.contador_reidentificacao >= config.FRAMES_REIDENTIFICAR:
                self.contador_reidentificacao = 0

                if self.rastreando_face:    #se ta rastreando pela face não identificada
                    self._reavaliar_rastreador_face(frame, img_gray)
                else:   # se esta rastreando pelo corpo ao invés de pela face
                    self._reavaliar_rastreador_body(frame, img_gray, x, y, w, h)

    def estado_identificando(self, frame, img_gray):    # se detectou, tentou detectar e conseguiu reconhecer, testa se por tempo suficiente reconhece mesmo
        distancia = self._prever_e_desenhar_debug(frame, img_gray)
        
        if distancia < config.LIMITE_CONFIANCA:
            tempo_identificado = time.time() - self.inicio_identificado
            if tempo_identificado > 2.0:    # ficou tempo suficiente reconheendo
                self.estado = config.EstadoPipeline.IMPRIME_IDENTIDADE
                cv2.putText(frame, 'IDENTIFICADO', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            else: # reconheceu mas ainda falta ficar mais empo reconhecendo
                self.estado = config.EstadoPipeline.IDENTIFICANDO
                cv2.putText(frame, 'Leonardo', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        else:   # não reconheceu, vai verificar se realmente é não reconhecido
            self.estado = config.EstadoPipeline.NAO_IDENTIFICANDO
            cv2.putText(frame, 'Nao Identificado', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            self.inicio_nao_identificado = time.time()

    def estado_nao_identificando(self, frame, img_gray):    #mesma logica que identificando mas para verificar se realmente e desconhecido
        distancia = self._prever_e_desenhar_debug(frame, img_gray)
        
        if distancia < config.LIMITE_CONFIANCA:
            self.estado = config.EstadoPipeline.IDENTIFICANDO
            self.inicio_identificado = time.time()
            cv2.putText(frame, 'Leonardo', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        else:
            tempo_nao_identificado = time.time() - self.inicio_nao_identificado
            if tempo_nao_identificado > 2.0:
                self.estado = config.EstadoPipeline.DETECTA_P_RASTREAR
                cv2.putText(frame, 'INICIANDO RASTREAMENTO', (30, config.ALTURA_TREINO // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            else:
                self.estado = config.EstadoPipeline.NAO_IDENTIFICANDO
                cv2.putText(frame, 'Nao Idenificado', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    def estado_imprime_identidade(self, frame, img_gray):
        distancia = self._prever_e_desenhar_debug(frame, img_gray)

        if distancia < config.LIMITE_CONFIANCA:
            self.inicio_perda_identificacao = None
            cv2.putText(frame, "IDENTIFICADO", (30, config.ALTURA_TREINO // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "ACESSO PERMITIDO", (20, config.ALTURA_TREINO // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            if self.inicio_perda_identificacao is None:
                self.inicio_perda_identificacao = time.time()
            tempo_perdido = time.time() - self.inicio_perda_identificacao

            if tempo_perdido > 2.2:
                self.inicio_nao_identificado = time.time()
                self.inicio_perda_identificacao = None
                self.estado = config.EstadoPipeline.NAO_IDENTIFICANDO
            else:
                cv2.putText(frame, "IDENTIFICADO", (30, config.ALTURA_TREINO // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "ACESSO PERMITIDO", (20, config.ALTURA_TREINO // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def estado_detecta_p_rastrear(self, frame, img_gray): # chamado quando o tracker perde o alvo e quando ou fica tempo suficiente não reconhecendo então detecta para começar a rastrear 
        cv2.putText(frame, "INICIANDO RASTREAMENTO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        deteccoes_face = self.detector_faces.detectMultiScale(img_gray, scaleFactor=1.02, minNeighbors=9)

        if len(deteccoes_face):
            self._iniciar_rastreamento_face(frame, tuple(deteccoes_face[0]))
            self.estado = config.EstadoPipeline.RASTREAMENTO
        else:
            deteccoes_body = self.detector_body.detectMultiScale(img_gray, scaleFactor=1.01, minNeighbors=9)
            if len(deteccoes_body):
                self._iniciar_rastreamento_body(frame, tuple(deteccoes_body[0]))
                self.estado = config.EstadoPipeline.RASTREAMENTO