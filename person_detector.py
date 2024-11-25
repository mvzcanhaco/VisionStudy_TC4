import torch
from ultralytics import YOLO


class PersonDetector:
    """Classe para detecção de pessoas em frames."""

    def __init__(self, detection_model, device, conf_threshold, iou_threshold):
        """
        Inicializa o detector de pessoas.

        Args:
            detection_model: Caminho ou nome do modelo de detecção YOLO.
            device: Dispositivo a ser usado ('cuda' ou 'cpu').
            conf_threshold: Limite de confiança para as detecções.
            iou_threshold: Limite de IOU para supressão não máxima.
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Inicializar YOLO
        self.model = YOLO(detection_model)
        if self.device == "cuda":
            self.model.to(self.device)

    def detect(self, frame):
        """
        Realiza a detecção de pessoas em um frame.

        Args:
            frame: Frame de imagem (numpy array).

        Returns:
            Lista de detecções, cada uma sendo um dicionário com 'bbox' e 'confidence'.
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # Apenas pessoas
        )

        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if conf >= self.conf_threshold:
                    detections.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})

        return detections
