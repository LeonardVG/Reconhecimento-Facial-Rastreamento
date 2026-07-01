from enum import Enum

# estados
class EstadoPipeline(Enum):
    DETECTA = "detecta"
    RECONHECIMENTO = "reconhecimento"
    DETECTA_BODY = "detecta_body"
    RASTREAMENTO = "rastreamento"
    IDENTIFICANDO = "identificando"
    NAO_IDENTIFICANDO = "nao_identificando"
    IMPRIME_IDENTIDADE = "imprime_identidade"
    DETECTA_P_RASTREAR = "detecta_p_rastrear"
#----------------------------------------------------------------------------------------
#caminhos
VIDEO_SOURCE = "test2.mp4"                                  # "http://192.168.1.4:8080/video" 
CASCADE_FACES_PATH = 'haarcascade_frontalface_default.xml'
CASCADE_BODY_PATH = 'fullbody.xml'
LBPH_MODEL_PATH = 'lbph_classifier_1classe_leonardo.yml'

# parâmetros e dimensões
LARGURA_TREINO = 320 
ALTURA_TREINO = 243
FRAMES_REIDENTIFICAR = 10
LIMITE_CONFIANCA = 68

DIR_OUTPUT = "capturas"         # Nome da pasta onde as imagens serão salvas
INTERVALO_SALVAMENTO = 4.0      # Tempo em segundos entre cada salvamento