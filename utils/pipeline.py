import cv2
import os
import time
from utils import config
from utils.estados import Estados

class PipelineReconhecimento(Estados):
    def __init__(self):
        # iniciando com as configurações do config.py
        self.webcam = cv2.VideoCapture(config.VIDEO_SOURCE)

        self.detector_faces = cv2.CascadeClassifier(config.CASCADE_FACES_PATH)
        self.detector_body = cv2.CascadeClassifier(config.CASCADE_BODY_PATH)

        self.lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
        self.lbph_classifier.read(config.LBPH_MODEL_PATH)

        self.tracker_face = cv2.TrackerCSRT_create()
        self.tracker_body = cv2.TrackerCSRT_create()

        self.rastreando_face = False
        self.rastreando_body = False

        self.contador_reidentificacao = 0

        self.inicio_perda_identificacao = None
        self.inicio_identificado = None
        self.inicio_nao_identificado = None

        self.estado = config.EstadoPipeline.DETECTA

        # ------ saalvamento de possiveis intrusos
        self.ultimo_salvamento = time.time()
        # cria a pasta de capturas automaticamente se não existir
        if not os.path.exists(config.DIR_OUTPUT):
            os.makedirs(config.DIR_OUTPUT)
            print(f"Pasta '{config.DIR_OUTPUT}' criada com sucesso.")

    def executar(self):
        """Loop principal do pipeline"""
        while True:
            ok, frame = self.webcam.read()
            if not ok: break

            frame = cv2.resize(frame, (config.LARGURA_TREINO, config.ALTURA_TREINO))
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Máquina de estados
            if self.estado == config.EstadoPipeline.DETECTA:
                self.estado_detecta(frame, img_gray)

            elif self.estado == config.EstadoPipeline.RECONHECIMENTO:
                self.estado_reconhecimento(frame, img_gray)

            elif self.estado == config.EstadoPipeline.DETECTA_BODY:
                self.estado_detecta_body(img_gray)

            elif self.estado == config.EstadoPipeline.RASTREAMENTO:
                self.estado_rastreamento(frame, img_gray)

            elif self.estado == config.EstadoPipeline.IDENTIFICANDO:
                self.estado_identificando(frame, img_gray)

            elif self.estado == config.EstadoPipeline.NAO_IDENTIFICANDO:
                self.estado_nao_identificando(frame, img_gray)

            elif self.estado == config.EstadoPipeline.IMPRIME_IDENTIDADE:
                self.estado_imprime_identidade(frame, img_gray)

            elif self.estado == config.EstadoPipeline.DETECTA_P_RASTREAR:
                self.estado_detecta_p_rastrear(frame, img_gray)

            cv2.imshow('Video', frame)

            key = cv2.waitKey(30)
            if key == 27: # ESC
                break

        self.webcam.release()
        cv2.destroyAllWindows()