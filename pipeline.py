import argparse
import csv
import subprocess
import sys
import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import shutil  # Importação adicionada

from activity_classifier import ActivityClassifier
from frames_extractor import ExtractorConfig, FrameExtractor
from sentiment_analyzer import HybridSentimentAnalyzer
from person_tracker import PersonTracker


@dataclass
class PersonData:
    person_id: int
    bbox: List[int]
    confidence: float
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    emotional_intensity: Optional[float] = None
    face_detected: bool = False
    face_bbox: Optional[List[int]] = None


@dataclass
class FrameData:
    frame_name: str
    activity: Optional[str] = None
    activity_confidence: Optional[float] = None
    persons: List[PersonData] = None


@dataclass
class PipelineConfig:
    video_path: Path
    fps: int
    execution_number: int
    detection_model: str
    track_model: str = "bytetrack.yaml"
    activity_model: str = "vit-b-32"
    skip_activity: bool = False
    skip_tracking: bool = False
    skip_sentiment: bool = False
    conf_threshold: float = 0.8
    iou_threshold: float = 0.3
    max_age: int = 30
    n_init: int = 3
    nn_budget: int = 100


def setup_logging(execution_number: int) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger('Pipeline')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / f'pipeline_exec_{execution_number}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_available_activity_models() -> dict:
    return {
        "vit-b-32": {
            "model_name": "ViT-B-32-quickgelu",
            "pretrained": "openai"
        },
        "vit-l-14": {
            "model_name": "ViT-L-14",
            "pretrained": "openai"
        },
        "vit-h-14": {
            "model_name": "ViT-H-14",
            "pretrained": "laion2b_s32b_b79k"
        },
        "convnext-large": {
            "model_name": "convnext_large",
            "pretrained": "laion2b_s29b_b131k_ft_soup"
        }
    }


def parse_arguments() -> PipelineConfig:
    parser = argparse.ArgumentParser(description='Video Processing Pipeline')

    parser.add_argument('--video',
                        default='data/tech_chall4.mp4',
                        type=str,
                        help='Path to the input video file')

    parser.add_argument('--fps',
                        default=2,
                        type=int,
                        help='Number of frames per second to extract')

    parser.add_argument('--exec-num',
                        default=1,
                        type=int,
                        help='Execution number for output organization')

    parser.add_argument('--detection-model',
                        default='yolo11s.pt',
                        type=str,
                        help='Path to YOLO model for detection and tracking')

    available_models = get_available_activity_models()
    parser.add_argument('--activity-model',
                        type=str,
                        choices=available_models.keys(),
                        default='vit-b-32',
                        help=f'Activity classification model to use. Available options: {", ".join(available_models.keys())}')

    parser.add_argument('--skip-activity',
                        action='store_true',
                        help='Skip activity classification step')

    parser.add_argument('--skip-tracking',
                        action='store_true',
                        help='Skip person tracking step')

    parser.add_argument('--skip-sentiment',
                        action='store_true',
                        help='Skip sentiment analysis step')

    parser.add_argument('--conf-threshold',
                        type=float,
                        default=0.6,
                        help='Confidence threshold for detections')

    parser.add_argument('--iou-threshold',
                        type=float,
                        default=0.4,
                        help='IOU threshold for tracking')

    parser.add_argument('--max-age',
                        type=int,
                        default=15,
                        help='Maximum number of frames to keep track alive')

    parser.add_argument('--n-init',
                        type=int,
                        default=1,
                        help='Number of frames for track initialization')

    parser.add_argument('--nn-budget',
                        type=int,
                        default=100,
                        help='Maximum size of feature budget')

    args = parser.parse_args()

    return PipelineConfig(
        video_path=Path(args.video),
        fps=args.fps,
        execution_number=args.exec_num,
        detection_model=args.detection_model,
        track_model="bytetrack.yaml",
        activity_model=args.activity_model,
        skip_activity=args.skip_activity,
        skip_tracking=args.skip_tracking,
        skip_sentiment=args.skip_sentiment,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
        n_init=args.n_init,
        nn_budget=args.nn_budget
    )


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging(config.execution_number)
        self.video_processor: Optional[FrameExtractor] = None
        self.activity_classifier: Optional[ActivityClassifier] = None
        self.person_tracker: Optional[PersonTracker] = None
        self.sentiment_analyzer: Optional[HybridSentimentAnalyzer] = None
        self.output_base_dir = Path(f"Outputs/Exec_{config.execution_number}")
        self.frame_data_list: List[FrameData] = []
        self.total_frames_in_video = 0
        self.total_frames_analyzed = 0
        self.total_frames_extracted = 0
        self.total_detections = 0
        self.total_people_tracked = set()
        self.activity_counts = {}
        self.sentiment_counts = {}

    def initialize_components(self):
        self.logger.info("Initializing pipeline components...")

        try:
            extractor_config = ExtractorConfig(
                video_path=self.config.video_path,
                execution_number=self.config.execution_number,
                fps=self.config.fps
            )
            self.video_processor = FrameExtractor(extractor_config, logger=self.logger)
            # Definir o template para nomes de frames com zeros à esquerda
            self.video_processor.frame_name_template = "frame%06d.png"

            self.logger.info(f"Initializing person detector with model: {self.config.detection_model}")
            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'


            if not self.config.skip_tracking:
                self.logger.info(f"Initializing person tracker")
                self.person_tracker = PersonTracker(self.config)

            if not self.config.skip_activity:
                activity_models = get_available_activity_models()
                model_config = activity_models[self.config.activity_model]

                self.logger.info(f"Initializing activity classifier with model: {self.config.activity_model}")
                self.activity_classifier = ActivityClassifier(
                    execution_number=self.config.execution_number,
                    model_name=model_config['model_name'],
                    pretrained=model_config['pretrained']
                )

            if not self.config.skip_sentiment:
                self.logger.info("Initializing sentiment analyzer")
                self.sentiment_analyzer = HybridSentimentAnalyzer()

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def process_frames(self):
        frames_path = Path(f"Outputs/Exec_{self.config.execution_number}/Frames_extracted")
        # Ordenar frames com base no número extraído do nome do arquivo
        frames = sorted(frames_path.glob("*.png"), key=lambda x: int(x.stem.replace('frame', '')))
        self.total_frames_analyzed = len(frames)

        self.logger.info(f"Starting to process {len(frames)} frames...")

        # Diretório para salvar os frames anotados
        annotated_frames_dir = self.output_base_dir / "annotated_frames"
        annotated_frames_dir.mkdir(parents=True, exist_ok=True)

        tracking_metrics = {
            'total_tracks': 0,
            'frames_processed': 0,
            'detections_per_frame': []
        }

        for idx, frame_path in enumerate(tqdm(frames, desc="Processing frames")):
            try:
                frame_pil = Image.open(frame_path).convert('RGB')
                frame_cv = cv2.imread(str(frame_path))

                if frame_cv is None:
                    self.logger.warning(f"Could not read frame: {frame_path}")
                    continue

                frame_name = frame_path.name

                frame_data = FrameData(frame_name=frame_name, persons=[])

                # Copiar o frame para anotação
                annotated_frame = frame_cv.copy()

                # Processar com tracker de pessoas (se não for pulado)
                track_info = {}
                if not self.config.skip_tracking:
                    _, track_info = self.person_tracker.process_frame(
                        frame_cv.copy(), frame_name
                    )

                    tracking_metrics['frames_processed'] += 1
                    num_detections = len(track_info)
                    tracking_metrics['detections_per_frame'].append(num_detections)
                    self.total_detections += num_detections

                    if hasattr(self.person_tracker, 'tracks_count'):
                        tracking_metrics['total_tracks'] = len(self.person_tracker.tracks_count)
                        self.total_people_tracked.update(self.person_tracker.tracks_count.keys())

                    for person_id, person_data in track_info.items():
                        bbox = person_data['bbox']
                        confidence = person_data['confidence']
                        x1, y1, x2, y2 = bbox
                        person_roi = frame_cv[y1:y2, x1:x2]
                        if person_roi.size == 0:
                            self.logger.debug(f"Empty ROI for person {person_id} in frame {frame_name}")
                            continue

                        person_entry = PersonData(
                            person_id=int(person_id),
                            bbox=bbox,
                            confidence=confidence
                        )

                        # Formatar person_id com zeros à esquerda
                        person_id_str = f"{person_id:03d}"

                        # Anotar bounding box da pessoa
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f'ID: {person_id_str}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                        if not self.config.skip_sentiment:
                            try:
                                self.logger.debug(f"Analyzing person {person_id} in frame {frame_name}")

                                analysis_result = self.sentiment_analyzer.analyze_image_array(person_roi)
                                self.logger.debug(f"Analysis result for person {person_id}: {analysis_result}")

                                summary = self.sentiment_analyzer.get_emotion_summary(analysis_result)
                                self.logger.debug(f"Emotion summary for person {person_id}: {summary}")

                                person_entry.face_detected = summary.get('face_count', 0) > 0

                                if person_entry.face_detected:
                                    person_entry.sentiment = summary['dominant_emotion']
                                    person_entry.sentiment_score = summary['dominant_score']
                                    person_entry.emotional_intensity = summary['emotional_intensity']

                                    face_bbox = analysis_result.get('face_bbox')
                                    if face_bbox:
                                        if isinstance(face_bbox[0], list):
                                            face_bbox = face_bbox[0]
                                        fx1, fy1, fx2, fy2 = face_bbox
                                        face_bbox_in_frame = [
                                            x1 + fx1,
                                            y1 + fy1,
                                            x1 + fx2,
                                            y1 + fy2
                                        ]
                                        person_entry.face_bbox = face_bbox_in_frame

                                        # Anotar bounding box da face
                                        cv2.rectangle(annotated_frame, (face_bbox_in_frame[0], face_bbox_in_frame[1]),
                                                      (face_bbox_in_frame[2], face_bbox_in_frame[3]),
                                                      (255, 0, 0), 2)
                                        # Anotar sentimento
                                        cv2.putText(annotated_frame, f'Sentiment: {person_entry.sentiment}',
                                                    (face_bbox_in_frame[0], face_bbox_in_frame[1] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

                                    self._save_person_frame_by_sentiment(person_roi, frame_name, person_id_str,
                                                                         person_entry.sentiment)
                                    sentiment = person_entry.sentiment
                                    if sentiment:
                                        self.sentiment_counts[sentiment] = self.sentiment_counts.get(sentiment, 0) + 1
                                else:
                                    self.logger.debug(
                                        f"No face detected for person {person_id} in frame {frame_name}")
                                    person_entry.face_detected = False

                            except Exception as e:
                                self.logger.debug(
                                    f"Error analyzing face for person {person_id} in frame {frame_name}: {str(e)}")
                                person_entry.face_detected = False

                        frame_data.persons.append(person_entry)

                # Processar com classificador de atividade (se não for pulado e houver track ID vinculado)
                if not self.config.skip_activity:
                    if track_info:  # Verifica se há track IDs associados no frame
                        activity, confidence, _ = self.activity_classifier.process_frame(
                            frame_pil, frame_name
                        )
                        self.logger.debug(f"Frame {frame_name} classified as {activity} with confidence {confidence}")

                        frame_data.activity = activity
                        frame_data.activity_confidence = confidence

                        # Anotar atividade no rodapé
                        cv2.putText(annotated_frame, f'Activity: {activity}',
                                    (10, annotated_frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                        if activity:
                            self.activity_counts[activity] = self.activity_counts.get(activity, 0) + 1
                    else:
                        self.logger.debug(
                            f"Skipping activity classification for frame {frame_name} as no track ID is present.")

                # Salvar o frame anotado com nome com zeros à esquerda
                annotated_frame_name = f"frame{idx + 1:06d}.png"
                annotated_frame_path = annotated_frames_dir / annotated_frame_name
                cv2.imwrite(str(annotated_frame_path), annotated_frame)

                self.frame_data_list.append(frame_data)

            except Exception as e:
                self.logger.error(f"Error processing frame {frame_path}: {str(e)}")

        if not self.config.skip_tracking and hasattr(self.person_tracker, 'get_tracking_stats'):
            self._save_tracking_metrics(tracking_metrics)

        self._save_data_to_csv()
        self._save_data_to_json()
        self._generate_summary(tracking_metrics)

        # Gerar o vídeo com os frames anotados
        self.generate_video_from_frames(annotated_frames_dir)

    def generate_video_from_frames(self, frames_dir: Path):
        """Gera um vídeo a partir dos frames anotados usando ffmpeg."""
        self.logger.info("Generating video from annotated frames with ffmpeg...")
        output_video_path = self.output_base_dir / f"annotated_video.mp4"

        # Verificar se existem frames na pasta
        frames = sorted(frames_dir.glob("*.png"), key=lambda x: int(x.stem.replace('frame', '')))
        if not frames:
            self.logger.error("No annotated frames found to generate video.")
            return

        # Certificar-se de que os frames estão nomeados sequencialmente com zeros à esquerda
        frame_template = "frame%06d.png"  # Usando 6 dígitos com zeros à esquerda
        temp_dir = frames_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for idx, frame_path in enumerate(frames):
            new_name = temp_dir / Path(frame_template % (idx + 1))  # Certificar-se de que o novo nome é um Path
            shutil.copy(str(frame_path), str(new_name))  # Copiar o frame para o diretório temporário

        # Gerar o vídeo usando ffmpeg
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # Sobrescrever o arquivo de saída se existir
                    "-framerate", str(self.video_processor.input_fps if self.video_processor else 30),
                    "-i", str(temp_dir / frame_template),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(output_video_path)
                ],
                check=True
            )
            self.logger.info(f"Annotated video saved at: {output_video_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to generate video with ffmpeg: {e}")
        finally:
            # Limpar a pasta temporária
            for temp_file in temp_dir.glob("*"):
                temp_file.unlink()
            temp_dir.rmdir()

    def _save_tracking_metrics(self, metrics: dict):
        self.logger.info("\n=== Tracking Metrics ===")
        self.logger.info(f"Total unique persons tracked: {metrics['total_tracks']}")
        self.logger.info(f"Total frames processed: {metrics['frames_processed']}")

        if len(metrics['detections_per_frame']) > 0:
            avg_detections = np.mean(metrics['detections_per_frame'])
            self.logger.info(f"Average detections per frame: {avg_detections:.2f}")

            stats = self.person_tracker.get_tracking_stats()
            metrics_path = self.output_base_dir / "tracking_metrics.csv"

            with open(metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Tracks', metrics['total_tracks']])
                writer.writerow(['Frames Processed', metrics['frames_processed']])
                writer.writerow(['Avg Detections per Frame', avg_detections])
                writer.writerow([])
                writer.writerow(['Track ID', 'Total Detections'])
                for track_id, detections in stats['detections_per_person'].items():
                    writer.writerow([track_id, detections])

    def _save_person_frame_by_sentiment(self, person_roi, frame_name: str, person_id_str: str, sentiment: str):
        sentiment_dir = self.output_base_dir / "Persons_by_sentiment" / sentiment
        sentiment_dir.mkdir(parents=True, exist_ok=True)
        person_frame_name = f"{frame_name}_person_{person_id_str}.png"
        output_path = sentiment_dir / person_frame_name
        cv2.imwrite(str(output_path), person_roi)

    def _save_data_to_csv(self):
        csv_path = self.output_base_dir / "complete_results.csv"
        fieldnames = [
            'frame_name', 'activity', 'activity_confidence', 'person_id', 'bbox',
            'confidence', 'face_detected', 'face_bbox', 'sentiment', 'sentiment_score',
            'emotional_intensity'
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for frame_data in self.frame_data_list:
                if frame_data.persons:
                    for person in frame_data.persons:
                        data_dict = {
                            'frame_name': frame_data.frame_name,
                            'activity': frame_data.activity,
                            'activity_confidence': frame_data.activity_confidence,
                            'person_id': f"{person.person_id:03d}",  # Formatar person_id com zeros à esquerda
                            'bbox': person.bbox,
                            'confidence': person.confidence,
                            'face_detected': person.face_detected,
                            'face_bbox': person.face_bbox,
                            'sentiment': person.sentiment,
                            'sentiment_score': person.sentiment_score,
                            'emotional_intensity': person.emotional_intensity
                        }
                        writer.writerow(data_dict)
                else:
                    data_dict = {
                        'frame_name': frame_data.frame_name,
                        'activity': frame_data.activity,
                        'activity_confidence': frame_data.activity_confidence,
                        'person_id': None,
                        'bbox': None,
                        'confidence': None,
                        'face_detected': None,
                        'face_bbox': None,
                        'sentiment': None,
                        'sentiment_score': None,
                        'emotional_intensity': None
                    }
                    writer.writerow(data_dict)

    def _save_data_to_json(self):
        json_path = self.output_base_dir / "complete_results.json"
        data_to_save = []
        for frame_data in self.frame_data_list:
            frame_dict = asdict(frame_data)
            if frame_data.persons:
                persons_list = []
                for person in frame_data.persons:
                    person_dict = asdict(person)
                    person_dict['bbox'] = person_dict['bbox']
                    person_dict['face_bbox'] = person_dict['face_bbox']
                    person_dict['person_id'] = f"{person.person_id:03d}"  # Formatar person_id com zeros à esquerda
                    persons_list.append(person_dict)
                frame_dict['persons'] = persons_list
            else:
                frame_dict['persons'] = []
            data_to_save.append(frame_dict)
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data_to_save, jsonfile, ensure_ascii=False, indent=4)

    def _generate_summary(self, tracking_metrics: dict):
        summary = {
            'total_frames_in_video': self.total_frames_in_video,
            'total_frames_extracted': self.total_frames_extracted,
            'total_frames_analyzed': self.total_frames_analyzed,
            'total_detections': self.total_detections,
            'total_people_tracked': len(self.total_people_tracked),
            'activity_counts': self.activity_counts,
            'sentiment_counts': self.sentiment_counts
        }

        summary_path = self.output_base_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(summary, jsonfile, ensure_ascii=False, indent=4)

    def run(self):
        try:
            self.logger.info("Starting pipeline execution...")

            # Passo 1: Extrair frames do vídeo
            self.logger.info("Step 1: Extracting frames from video...")
            extracted_frames = self.video_processor.extract_frames()
            self.total_frames_in_video = self.video_processor.total_frames
            self.total_frames_extracted = extracted_frames
            self.logger.info("Frame extraction completed successfully")

            # Passo 2: Processar todos os frames
            self.logger.info("Step 2: Processing frames...")
            self.process_frames()
            self.logger.info("Frame processing completed successfully")

            self.logger.info("Pipeline execution completed successfully")

        except Exception as e:
            self.logger.error(f"Error during pipeline execution: {str(e)}")
            raise



