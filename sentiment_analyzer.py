import cv2
import numpy as np
from deepface import DeepFace
import torch
import logging
import tempfile
from typing import List, Dict, Any, Optional

# Configuração do logging
logging.basicConfig(level=logging.INFO)


class HybridSentimentAnalyzer:
    """
    Classe para analisar sentimentos em imagens combinando análise facial.
    """

    def __init__(self):
        """
        Inicializa o HybridSentimentAnalyzer.
        """
        # Definir etiquetas de emoções consistentes
        self.emotion_labels = ['happy', 'sad', 'angry', 'surprise', 'neutral']

    def analyze_image_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Analisa uma imagem (array numpy) usando o DeepFace.
        """
        logging.debug(f"Analyzing image array with shape: {image_array.shape}")

        # Verificar se a imagem é grande o suficiente
        height, width = image_array.shape[:2]
        if height < 48 or width < 48:
            logging.debug(f"Image too small for face detection: {width}x{height}")
            return {
                'success': False,
                'error': 'Image too small for face detection',
                'face_count': 0,
                'emotions': None,
                'face_bbox': None
            }

        # Chamar _analyze_faces_array com a imagem no formato BGR
        results = {
            'facial': self._analyze_faces_array(image_array)
        }
        combined_results = self._combine_results(results)
        return combined_results

    def _analyze_faces_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Utiliza o DeepFace para análise de emoções faciais em um array numpy.
        """
        try:
            # Salvar a imagem temporariamente
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_image_file:
                cv2.imwrite(temp_image_file.name, image_array)

                # Passar o caminho do arquivo para o DeepFace
                analysis = DeepFace.analyze(
                    img_path=temp_image_file.name,
                    actions=['emotion'],
                    enforce_detection=True,
                    detector_backend='retinaface'

                )

            logging.debug(f"DeepFace analysis output: {analysis}")

            if isinstance(analysis, list):
                # Múltiplos rostos detectados
                combined_emotions = {}
                face_regions = []
                for face in analysis:
                    for emotion, value in face['emotion'].items():
                        combined_emotions[emotion] = combined_emotions.get(emotion, 0) + value
                    face_regions.append([
                        face['region']['x'],
                        face['region']['y'],
                        face['region']['x'] + face['region']['w'],
                        face['region']['y'] + face['region']['h']
                    ])

                # Normaliza as emoções
                normalized_emotions = self._normalize_emotions(combined_emotions)
                return {
                    'success': True,
                    'face_count': len(analysis),
                    'emotions': normalized_emotions,
                    'face_bbox': face_regions
                }
            else:
                # Um único rosto detectado
                normalized_emotions = self._normalize_emotions(analysis['emotion'])
                face_region = analysis['region']
                face_bbox = [
                    face_region['x'],
                    face_region['y'],
                    face_region['x'] + face_region['w'],
                    face_region['y'] + face_region['h']
                ]
                return {
                    'success': True,
                    'face_count': 1,
                    'emotions': normalized_emotions,
                    'face_bbox': face_bbox
                }
        except Exception as e:
            if "Face could not be detected" in str(e):
                logging.debug(f"Nenhuma face detectada na imagem.")
            else:
                logging.error(f"Erro inesperado na análise facial: {e}")
            return {
                'success': False,
                'error': str(e),
                'face_count': 0,
                'emotions': None,
                'face_bbox': None
            }

    def _normalize_emotions(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Normaliza as pontuações de emoções para somarem 1.
        """
        total = sum(emotions.values())
        if total > 0:
            return {k: v / total for k, v in emotions.items()}
        else:
            return emotions

    def _combine_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combina os resultados da análise facial.
        """
        combined_analysis = {
            'face_count': results['facial'].get('face_count') if results['facial']['success'] else 0,
            'face_bbox': results['facial'].get('face_bbox') if results['facial']['success'] else None,
            'emotions': results['facial'].get('emotions') if results['facial']['success'] else None
        }

        return combined_analysis

    def calculate_emotional_intensity(self, results: Dict[str, Any]) -> float:
        """
        Calcula a intensidade emocional geral.
        """
        if 'emotions' in results and results['emotions']:
            emotions = np.array(list(results['emotions'].values()))
            emotions = emotions[emotions > 0]
            if len(emotions) > 0:
                entropy = -np.sum(emotions * np.log2(emotions + 1e-10))
                intensity = 1 - (entropy / np.log2(len(self.emotion_labels)))
                return intensity
        return 0.0

    def get_dominant_emotion(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retorna a emoção dominante e sua pontuação.
        """
        if 'emotions' in results and results['emotions']:
            dominant = max(results['emotions'].items(), key=lambda x: x[1])
            return {
                'emotion': dominant[0],
                'score': dominant[1]
            }
        return None

    def get_emotion_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retorna um resumo da análise emocional.
        """
        if not results.get('emotions'):
            return {"summary": "Nenhuma emoção detectada"}

        dominant = self.get_dominant_emotion(results)
        intensity = self.calculate_emotional_intensity(results)

        summary = {
            'dominant_emotion': dominant['emotion'] if dominant else None,
            'dominant_score': dominant['score'] if dominant else None,
            'emotional_intensity': intensity,
            'face_count': results.get('face_count', 0),
        }
        return summary
