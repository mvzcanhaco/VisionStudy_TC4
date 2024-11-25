import torch
from pathlib import Path
import cv2
import logging
from typing import Dict, List, Tuple
from ultralytics import YOLO
import numpy as np


class PersonTracker:
    """Classe para detecção e tracking de pessoas em frames usando StrongSort."""

    def __init__(self, config):
        """
        Inicializa o tracker de pessoas usando StrongSort.

        Args:
            config: Configuração da pipeline contendo todos os parâmetros necessários
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.execution_number = config.execution_number
        self.conf_threshold = config.conf_threshold
        self.iou_threshold = config.iou_threshold

        # Parâmetros adicionais do StrongSort
        self.track_thresh = 0.6  # Limiar de confiança para início de uma nova track
        self.match_thresh = 0.9  # Limiar para associação de tracks
        self.track_buffer = 15  # Buffer para perda temporária de detecção

        # Configurar diretórios de saída
        self.output_dir = Path(f"Outputs/Exec_{config.execution_number}")
        self.tracks_dir = self.output_dir / "Person_Tracks"
        self.annotated_dir = self.output_dir / "Annotated_Frames"

        self.tracks_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)

        # Configuração de logging
        self.logger = self._setup_logger()

        try:
            # Inicializar YOLO com StrongSort
            self.model = YOLO(config.detection_model)
            self.model.fuse()  # Otimizar o modelo
            self.model.to(self.device)
            self.model.overrides['tracker'] = config.track_model  # Usar StrongSort
            self.model.overrides['device'] = self.device
            self.model.overrides['conf'] = self.conf_threshold
            self.model.overrides['iou'] = self.iou_threshold
            self.model.overrides['classes'] = [0]  # Classe pessoa
            self.model.overrides['agnostic_nms'] = True
            self.model.overrides['max_det'] = 1000

            # # Congelar camadas não necessárias
            # self._freeze_layers()

            self.logger.info(f"Modelo YOLO inicializado com StrongSort no dispositivo {self.device}")

        except Exception as e:
            self.logger.error(f"Erro ao inicializar os modelos: {str(e)}", exc_info=True)
            raise e  # Propagar a exceção para tratamento upstream

        # Dicionário para controle de tracks
        self.tracks_count = {}

    def _setup_logger(self) -> logging.Logger:
        """Configura o logger específico para o tracker."""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.execution_number}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)

            log_path = self.tracks_dir / "tracking.log"
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    def _save_person_crop(self, frame: np.ndarray, bbox: List[int], track_id: int, frame_name: str) -> None:
        """Salva o recorte da pessoa detectada."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                return

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return

            # Filtrar detecções com tamanhos não realistas
            crop_area = (x2 - x1) * (y2 - y1)
            frame_area = frame.shape[0] * frame.shape[1]
            area_ratio = crop_area / frame_area
            if area_ratio < 0.001 or area_ratio > 0.5:
                # Ignorar detecções muito pequenas ou que cobrem mais da metade do frame
                return

            # Criar diretório para o ID
            person_dir = self.tracks_dir / f"ID_{track_id:03d}"
            person_dir.mkdir(exist_ok=True)

            # Nome do arquivo baseado no frame original
            frame_number = frame_name.split('.')[0]
            output_path = person_dir / f"bbox_{frame_number}.jpg"

            cv2.imwrite(str(output_path), person_crop)
            self.tracks_count[track_id] = self.tracks_count.get(track_id, 0) + 1

        except cv2.error as e:
            self.logger.error(f"Erro do OpenCV ao salvar recorte da pessoa: {str(e)}")
        except Exception as e:
            self.logger.error(f"Erro inesperado ao salvar recorte da pessoa: {str(e)}", exc_info=True)

    def process_frame(self, frame: np.ndarray, frame_name: str) -> Tuple[np.ndarray, Dict]:
        """Processa um frame para detecção e tracking de pessoas usando StrongSort."""
        try:
            # Aplicar pré-processamento de imagem
            preprocessed_frame = self._preprocess_frame(frame)

            # Detecção e tracking com YOLO e StrongSort
            results = self.model.track(
                source=preprocessed_frame,
                persist=True,
                verbose=False,
                stream=False,
                tracker='bytetrack.yaml',
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                agnostic_nms=False,
                device=self.device
            )

            if not results or len(results) == 0:
                return frame, {}

            result = results[0]
            if not result.boxes:
                return frame, {}

            annotated_frame = frame

            track_info = {}

            for box in result.boxes:
                if box.id is None or box.conf is None:
                    continue

                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Salvar recorte da pessoa com filtragem
                self._save_person_crop(frame, [x1, y1, x2, y2], track_id, frame_name)

                # Anotar frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"ID_{track_id:03d}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                track_info[track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'frame': frame_name
                }

            # Salvar frame anotado
            cv2.imwrite(str(self.annotated_dir / frame_name), annotated_frame)

            return annotated_frame, track_info

        except cv2.error as e:
            self.logger.error(f"Erro do OpenCV ao processar frame: {str(e)}")
            return frame, {}
        except Exception as e:
            self.logger.error(f"Erro inesperado ao processar frame: {str(e)}", exc_info=True)
            return frame, {}

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Aplica técnicas de pré-processamento para melhorar a detecção."""
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicar equalização de histograma
        equalized = cv2.equalizeHist(gray)
        # Converter de volta para BGR
        preprocessed_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        return preprocessed_frame

    def get_tracking_stats(self) -> Dict:
        """Retorna estatísticas do tracking."""
        return {
            'total_unique_persons': len(self.tracks_count),
            'total_detections': sum(self.tracks_count.values()),
            'detections_per_person': {
                f"ID_{track_id:03d}": count
                for track_id, count in self.tracks_count.items()
            }
        }
