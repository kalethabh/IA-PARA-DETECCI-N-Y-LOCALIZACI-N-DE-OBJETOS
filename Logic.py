import cv2
import numpy as np

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    print("Error: PyTorch no está instalado. Instálalo con 'pip install torch torchvision torchaudio'")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    # Verificación de dispositivo (CPU/GPU)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Definimos la arquitectura del agente
    class AgenteDeteccion:
        """Agente inteligente basado en aprendizaje profundo para detección de objetos."""
        
        def __init__(self):
            """Inicializa el agente con una red neuronal preentrenada."""
            self.modelo = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.modelo.to(dispositivo)
            self.modelo.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((800, 800)),
                transforms.ToTensor()
            ])
            self.umbral_confianza = 0.7
        
        def percibir(self, imagen):
            """Capta la imagen y la transforma para el modelo de detección."""
            imagen_transformada = self.transform(imagen)
            return imagen_transformada.unsqueeze(0).to(dispositivo)
        
        def procesar(self, imagen):
            """Procesa la imagen y detecta objetos."""
            with torch.no_grad():
                prediccion = self.modelo(imagen)
            return prediccion
        
        def decidir(self, prediccion):
            """Decide qué objetos reportar basándose en la confianza del modelo."""
            objetos_detectados = []
            for idx, puntaje in enumerate(prediccion[0]['scores']):
                if puntaje > self.umbral_confianza:
                    objetos_detectados.append({
                        'clase': prediccion[0]['labels'][idx].item(),
                        'caja': prediccion[0]['boxes'][idx].tolist(),
                        'confianza': puntaje.item()
                    })
            return objetos_detectados
        
        def actuar(self, detecciones):
            """Ejecuta acciones según la detección realizada."""
            if detecciones:
                print("Alerta: Se han detectado objetos.")
                for obj in detecciones:
                    print(f"Objeto detectado: {obj['clase']} con confianza {obj['confianza']:.2f}")
            else:
                print("No se han detectado objetos relevantes.")

    # Función principal para capturar imagen y ejecutar el agente
    def ejecutar_agente():
        """Captura una imagen de la webcam y ejecuta el proceso del agente."""
        agente = AgenteDeteccion()
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen_tensor = agente.percibir(imagen)
            prediccion = agente.procesar(imagen_tensor)
            detecciones = agente.decidir(prediccion)
            agente.actuar(detecciones)
        else:
            print("Error al capturar la imagen.")

    # Ejecutar el agente
    if __name__ == "__main__":
        ejecutar_agente()
