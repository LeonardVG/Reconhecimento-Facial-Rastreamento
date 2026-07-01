# Pipeline de Detecção, Reconhecimento e Rastreamento com OpenCV utilizando Técnicas Clássicas

Este repositório contém um sistema de visão computacional desenvolvido em Python com OpenCV. O objetivo do sistema é detectar faces ou corpos, reconhecer identidades conhecidas e manter o rastreamento contínuo dos indivíduos em vídeo.

O modelo de reconhecimento facial utilizado (**LBPH**) foi [**previamente treinado**](treinando_lbph_git.ipynb) utilizando o ambiente do Google Colab e importado para este pipeline local.

---

## Arquitetura e Modularização

O código foi construído com base em princípios de Programação Orientada a Objetos (POO), utilizando o padrão de projeto estrutural Mixin para separar as responsabilidades. O projeto está dividido em quatro arquivos principais:

* **`config.py`**: Centraliza todas as constantes numéricas, caminhos de arquivos (cascades, modelos, vídeos) e definições dos enumeradores (Enum) para a máquina de estados.
* **`estados.py`**: Contém a classe `Estados`. Este arquivo isola toda a lógica específica de manipulação de imagem, atualização de rastreadores e métodos auxiliares para cada um dos estados do sistema.
* **`pipeline.py`**: Define a classe principal `PipelineReconhecimento`, que herda de `Estados`. É responsável pela inicialização do sistema e pelo loop principal (captura de vídeo).
* **`main.py`**: Ponto de entrada do programa. Apenas instancia o pipeline e executa o loop principal.

* **`criar.py`**: script para criação do dataset de treinamento e/ou teste do reconhecedor LBPH.

---

## Conceito de Máquina de Estados

O controle de fluxo do pipeline não depende de condicionais aninhadas complexas, mas sim de uma **Máquina de Estados Finita (FSM)**. O sistema se encontra em um único estado por vez e transita para o próximo estado com base nos resultados do processamento do frame atual. 

Os estados disponíveis são:

* **DETECTA**: Estado inicial. Busca passivamente por rostos utilizando Haar Cascades. Se encontrar, avança para o reconhecimento. Se não encontrar, tenta detectar um corpo.
* **DETECTA_BODY**: Busca por corpos caso rostos não estejam visíveis no momento. Se encontrar, inicia o rastreamento.
* **RECONHECIMENTO**: Avalia a face detectada contra o modelo LBPH.
* **IDENTIFICANDO / NAO_IDENTIFICANDO**: Estados de transição baseados no tempo. Validam se a previsão do modelo LBPH se mantém consistente por alguns segundos, evitando falsos positivos e oscilações na interface.
* **IMPRIME_IDENTIDADE**: Estado de sucesso estabilizado. Mantém o reconhecimento validado e exibe a permissão de acesso na tela.
* **RASTREAMENTO**: Utiliza o rastreador CSRT para seguir o rosto ou corpo de forma performática. Periodicamente, o sistema tenta reavaliar o alvo e atualizar os rastreadores.
* **DETECTA_P_RASTREAR**: Estado de recuperação (fallback). Acionado se o rastreador falhar ou perder o alvo, forçando uma nova detecção limpa em toda a imagem para reiniciar o processo.

---

## Funcionalidades Principais

* **Detecção Dupla**: Utiliza classificadores Haar Cascade tanto para face frontal (`haarcascade_frontalface_default.xml`) quanto para corpo inteiro (`fullbody.xml`).
* **Reconhecimento Facial (LBPH)**: Avalia distâncias de confiança e aplica regras de tolerância temporal para atestar a identidade de forma segura.
* **Rastreamento CSRT**: Substitui a necessidade de rodar a detecção (processo pesado) em todos os frames, atualizando a posição do objeto rastreado dinamicamente.
* **Log de Capturas**: Durante o estado de rastreamento, o sistema salva automaticamente uma cópia do frame completo e um recorte (crop/bbox) do alvo a cada intervalo de tempo estipulado, armazenando no diretório definido em configuração.

---

## Requisitos e Execução

### Dependências
Certifique-se de possuir o Python instalado e instale a biblioteca do OpenCV adaptada para contribuições (que inclui o LBPH e os rastreadores):

```bash
pip install opencv-contrib-python numpy
```
## Execução
* Garanta que os arquivos `.xml` dos classificadores Haar e o arquivo `.yml` do modelo LBPH gerado no Colab estejam na mesma raiz do projeto.

* Altere o caminho do vídeo de entrada na variável VIDEO_SOURCE dentro do arquivo config.py (ou defina como 0 para usar a webcam).

* Execute o comando no terminal:
  ```bash
  python main.py
  ```
* Para interromper o programa a qualquer momento, selecione a janela do vídeo e pressione a tecla ESC.
