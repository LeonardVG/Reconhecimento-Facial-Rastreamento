from utils.pipeline import PipelineReconhecimento

if __name__ == "__main__":
    print("Iniciando o sistema de reconhecimento...")

    app = PipelineReconhecimento()
    
    # roda o loop principal
    app.executar()