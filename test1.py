# Importações necessárias
import os
import gc
import cv2
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Union, Optional
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
from fer import FER
import time

print("PyTorch version:", torch.__version__)
print("Is MPS available in PyTorch?", torch.backends.mps.is_available())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
    force=True
)

# Constantes
SEQUENCE_LENGTH = 8
MAX_DIMENSION = 320
BATCH_SIZE = 8
BUFFER_SIZE = 30
SKIP_FRAMES = 10
TARGET_SIZE = (256, 256)
BoundingBox = Tuple[int, int, int, int]

activity_labels = [
    "people sitting and reading",
    "smiling",
    "ballet dancing",
    "standing and looking at camera",
    "expressing surprise",
    "laughing",
    "lying down and sleeping",
    "making funny faces",
    "medical consultation",
    "group working on computers",
    "using cell phone",
    "using computer",
    "group talking",
    "anomaly or undefined",
    "handshaking",
]

# Dicionário com descrições detalhadas para atividades específicas
ACTIVITY_DESCRIPTIONS = {
    "Group of People": [
        "a group of people sitting around a table",
        "a group of people standing close to each other",
        "people gathered around a table together",
        "individuals engaged in a group discussion",
        "people working together on computers",
        "a group of people talking with each other",
        "colleagues collaborating in an office setting",
        "friends sitting and chatting together",
        "people sitting with heads down reading",
        "people standing and talking to each other in a group",
        "group of friends having a conversation",
    ],
    "smiling": [
        "a person smiling at the camera",
        "someone smiling warmly",
        "individual with a happy smile",
        "person showing a big smile",
        "smiling person looking at camera"
    ],
    "ballet dancing": [
        "a ballerina dancing gracefully",
        "person performing ballet dance",
        "someone dancing ballet on stage",
        "ballet dancer in motion",
        "graceful ballet performance"
    ],
    "standing and looking at camera": [
        "a man standing and looking at the camera",
        "person standing still facing the camera",
        "individual gazing directly at camera",
        "man standing and staring at viewer",
        "person standing and making eye contact with camera"
    ],
    "expressing surprise": [
        "a person expressing surprise at the camera",
        "someone with a surprised facial expression",
        "individual showing surprise",
        "person reacting with astonishment",
        "surprised person looking at camera"
    ],
    "laughing": [
        "a person laughing",
        "someone laughing out loud",
        "individual enjoying a joke",
        "person smiling broadly and laughing",
        "laughing person in a happy mood"
    ],
    "lying down and sleeping": [
        "a woman lying on a sofa yawning",
        "person lying down and falling asleep",
        "woman yawning and sleeping on couch",
        "individual lying on sofa and yawning",
        "sleeping person on a couch"
    ],
    "making funny faces": [
        "a man making funny faces at the camera",
        "person making faces and expressions",
        "individual pulling silly faces at camera",
        "man making goofy expressions",
        "person making faces directly to camera"
    ],
    "medical consultation": [
        "a doctor and patient in a medical consultation",
        "doctor talking to patient",
        "person recovering in a hospital room",
        "surgical operation in progress",
        "doctor and nurse in an operating room",
        "bandaged patients recovering",
        "people in surgery gowns in a medical setting",
        "patients resting post-surgery",
        "individuals in a recovery room"
    ],
    "using cell phone": [
        "person using a cell phone",
        "someone texting on smartphone",
        "individual looking at mobile phone",
        "person browsing on phone",
        "using a smartphone device"
    ],
    "using computer": [
        "person using a computer",
        "someone typing on a keyboard",
        "individual working on a laptop",
        "person focused on computer screen",
        "using a desktop computer"
    ],
    "Reading": [
        "a person reading a book",
        "people sitting and reading quietly",
        "individual reading a document attentively",
        "someone reading on a tablet or phone",
        "group of people reading together in silence",
        "a person focused on reading",
        "people gathered and reading books"
    ],
    "anomaly or undefined": [
        "undefined or unclear action",
        "anomaly detected with unusual features",
        "simulated entity",
        "necromancy or supernatural",
        "uncertain classification in image",
        "detected object appears abnormal",
        "undefined or unknown activity",
        "unusual or unexpected behavior",
        "undefined or unclassified object",
        "anomaly detected in image",
        "unexpected or unexplained action",
        "unreal or supernatural entity",
    ],
}


def generate_clip_prompts(action: str) -> List[str]:
    """
    Gera prompts formatados para o CLIP baseado na ação.
    """
    base_prompts = ACTIVITY_DESCRIPTIONS.get(action, [f"a person {action}"])
    enhanced_prompts = []

    for prompt in base_prompts:
        # Adicionar variações de contexto
        enhanced_prompts.extend([
            prompt,
            f"photo of {prompt}",
            f"image of {prompt}",
            f"picture of {prompt}",
            f"shot of {prompt}",
            f"video frame of {prompt}",
            f"snapshot of {prompt}"
        ])

    return enhanced_prompts


def ensure_numpy_array(frame: Union[np.ndarray, torch.Tensor]) -> Optional[np.ndarray]:
    try:
        if isinstance(frame, torch.Tensor):
            frame = frame.numpy()
        if not isinstance(frame, np.ndarray):
            return None
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        return frame
    except Exception as e:
        logging.error(f"Error converting frame: {str(e)}")
        return None


def preprocess_frame_fixed(
        frame: Union[np.ndarray, torch.Tensor],
        target_size: Tuple[int, int] = TARGET_SIZE
) -> Optional[torch.Tensor]:
    try:
        if frame is None:
            return None
        frame_np = ensure_numpy_array(frame)
        if frame_np is None:
            return None
        if len(frame_np.shape) != 3:
            return None
        frame_resized = cv2.resize(frame_np, target_size, interpolation=cv2.INTER_AREA)
        if frame_resized.shape[-1] != 3:
            if len(frame_resized.shape) == 2:
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
            else:
                return None
        frame_tensor = torch.from_numpy(frame_resized).float()
        frame_tensor = frame_tensor / 255.0
        return frame_tensor
    except Exception as e:
        logging.error(f"Error preprocessing frame: {str(e)}")
        logging.debug(f"Frame shape: {frame.shape if hasattr(frame, 'shape') else 'unavailable'}")
        return None


# Classe CLIPActivityValidator
class CLIPActivityValidator:
    def __init__(self, activity_labels: List[str]):
        # Ajuste para usar o backend MPS se disponível
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.activity_labels = activity_labels
        self.activity_embeddings = self._precompute_activity_embeddings()
        self.prompts_per_activity = self._count_prompts_per_activity()

    def _count_prompts_per_activity(self) -> int:
        # Conta quantos prompts são gerados para cada atividade
        sample_action = self.activity_labels[0]
        sample_prompts = generate_clip_prompts(sample_action)
        return len(sample_prompts)

    def _precompute_activity_embeddings(self) -> torch.Tensor:
        activity_descriptions = []
        for activity in self.activity_labels:
            prompts = generate_clip_prompts(activity)
            activity_descriptions.extend(prompts)
        with torch.no_grad():
            text_inputs = self.processor(
                text=activity_descriptions,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            text_embeddings = self.model.get_text_features(**text_inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            return text_embeddings.cpu()

    @torch.no_grad()
    def get_clip_scores(self, frame: np.ndarray) -> torch.Tensor:
        inputs = self.processor(
            images=frame,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        image_embeddings = self.model.get_image_features(**inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        similarity = image_embeddings.cpu() @ self.activity_embeddings.t()
        return similarity.squeeze()


# Classe CLIPClassifier
class CLIPClassifier:
    def __init__(
            self,
            activity_labels: List[str],
            target_size: Tuple[int, int] = TARGET_SIZE
    ):
        self.activity_labels = activity_labels
        self.target_size = target_size
        self.clip_validator = CLIPActivityValidator(activity_labels)
        self.frame_queues = defaultdict(lambda: deque(maxlen=1))
        self.prediction_cache = {}
        self.cache_timeout = 5
        self.frame_counter = defaultdict(int)
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess_frame(self, frame: Union[np.ndarray, torch.Tensor]) -> Optional[torch.Tensor]:
        try:
            if hasattr(frame, 'shape'):
                self.logger.debug(f"Original frame shape: {frame.shape}")
            processed = preprocess_frame_fixed(frame, self.target_size)
            if processed is not None:
                self.logger.debug(f"Processed frame shape: {processed.shape}")
            else:
                self.logger.warning("Failed to process frame")
            return processed
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return None

    def update_frames(self, track_id: int, frame: Union[np.ndarray, torch.Tensor]) -> None:
        try:
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is not None:
                expected_shape = (self.target_size[0], self.target_size[1], 3)
                actual_shape = tuple(processed_frame.shape)
                if actual_shape != expected_shape:
                    self.logger.warning(
                        f"Incorrect shape for track {track_id}: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
                    return
                self.frame_queues[track_id].append(processed_frame)
                self.frame_counter[track_id] += 1
                if self.frame_counter[track_id] % self.cache_timeout == 0:
                    if track_id in self.prediction_cache:
                        del self.prediction_cache[track_id]
        except Exception as e:
            self.logger.error(f"Error updating frames for track {track_id}: {str(e)}")

    def get_activity(self, track_id: int) -> Tuple[str, float]:
        if track_id in self.prediction_cache:
            return self.prediction_cache[track_id]
        try:
            frame = self.frame_queues[track_id][0]
            frame_np = frame.numpy()
            clip_scores = self.clip_validator.get_clip_scores(frame_np)
            clip_probs = torch.softmax(clip_scores, dim=0)
            final_idx = torch.argmax(clip_probs).item()
            final_confidence = clip_probs[final_idx].item()
            # Ajustar o índice considerando múltiplos prompts por atividade
            activity_idx = final_idx // self.clip_validator.prompts_per_activity
            activity = self.activity_labels[activity_idx]
            self.prediction_cache[track_id] = (activity, final_confidence)
            return activity, final_confidence
        except Exception as e:
            logging.error(f"Error getting activity for track {track_id}: {str(e)}")
            return "unknown", 0.0

    def cleanup_old_tracks(self, active_tracks: List[int]) -> None:
        current_tracks = set(self.frame_queues.keys())
        inactive_tracks = current_tracks - set(active_tracks)
        for track_id in inactive_tracks:
            del self.frame_queues[track_id]
            if track_id in self.prediction_cache:
                del self.prediction_cache[track_id]
            if track_id in self.frame_counter:
                del self.frame_counter[track_id]

    def classify_frame(self, frame: np.ndarray) -> List[Tuple[str, float]]:
        """
        Classifica o frame inteiro e retorna as 2 atividades com maiores scores.
        """
        try:
            frame_np = ensure_numpy_array(frame)
            if frame_np is None:
                return [("unknown", 0.0), ("unknown", 0.0)]
            # Preprocessar o frame conforme necessário pelo CLIP
            inputs = self.clip_validator.processor(
                images=frame_np,
                return_tensors="pt",
                padding=True
            ).to(self.clip_validator.device)
            with torch.no_grad():
                image_embeddings = self.clip_validator.model.get_image_features(**inputs)
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                similarity = image_embeddings.cpu() @ self.clip_validator.activity_embeddings.t()
                similarity = similarity.squeeze()
                probs = torch.softmax(similarity, dim=0)
                top_probs, top_idxs = torch.topk(probs, k=2)
                top_activities = []
                for idx, prob in zip(top_idxs, top_probs):
                    activity_idx = idx // self.clip_validator.prompts_per_activity
                    activity = self.activity_labels[activity_idx]
                    top_activities.append((activity, prob.item()))
                return top_activities
        except Exception as e:
            self.logger.error(f"Error classifying frame: {str(e)}")
            return [("unknown", 0.0), ("unknown", 0.0)]


# Classe PersonTracker
class PersonTracker:
    def __init__(self, max_disappeared=10, min_iou=0.5):
        self.next_id = 0
        self.trackers = {}
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.frame_count = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def calculate_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def calculate_iou(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1_p, y1_p, x2_p, y2_p = bbox2
        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou if iou > 0 else 0

    def update(self, detections):
        self.frame_count += 1
        current_trackers = {}
        if not detections:
            self.logger.info("No detections found in this frame.")
            return {}
        if self.trackers:
            iou_matrix = np.zeros((len(detections), len(self.trackers)))
            for i, bbox in enumerate(detections):
                for j, (track_id, data) in enumerate(self.trackers.items()):
                    iou_matrix[i, j] = self.calculate_iou(bbox, data['bbox'])
            while np.any(iou_matrix > self.min_iou):
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                track_id = list(self.trackers.keys())[j]
                current_trackers[track_id] = {
                    'bbox': detections[i],
                    'last_seen': self.frame_count
                }
                self.logger.info(
                    f"Associated ID {track_id} with detection {detections[i]} (IoU={iou_matrix[i, j]:.2f})")
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0
        used_detections = set(tuple(d['bbox']) for d in current_trackers.values())
        for bbox in detections:
            if tuple(bbox) not in used_detections:
                current_trackers[self.next_id] = {
                    'bbox': bbox,
                    'last_seen': self.frame_count
                }
                self.logger.info(f"New person detected: assigned ID {self.next_id} with bbox {bbox}")
                self.next_id += 1
        inactive_ids = []
        for track_id, data in self.trackers.items():
            if track_id not in current_trackers:
                if self.frame_count - data['last_seen'] > self.max_disappeared:
                    inactive_ids.append(track_id)
        for track_id in inactive_ids:
            self.logger.info(f"Removing ID {track_id} due to inactivity.")
            del self.trackers[track_id]
        self.trackers.update(current_trackers)
        return current_trackers


def optimize_frame(frame):
    if frame is None:
        return None
    height, width = frame.shape[:2]
    if height > MAX_DIMENSION or width > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(height, width)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return frame


def analyze_sentiment_safe(face_img):
    sentiments = []
    try:
        if face_img is None or face_img.size == 0:
            logging.warning("Invalid face image for sentiment analysis.")
            return ["unknown"]

        # Initialize the FER detector
        detector = FER(mtcnn=True)

        # Analyze the face image
        results = detector.detect_emotions(face_img)

        if not results:
            return ["unknown"]

        # Extract the dominant emotion
        emotions = results[0]["emotions"]
        sentiment = max(emotions, key=emotions.get)
        sentiments.append(sentiment)
        logging.info(f"Sentiment Analysis: {sentiment}")
        return sentiments
    except Exception as e:
        logging.warning(f"Error in sentiment analysis: {e}")
        return ["unknown"]


def process_person_crop(frame, bbox: BoundingBox, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    try:
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
        scale = min(target_size[0] / person_crop.shape[0], target_size[1] / person_crop.shape[1])
        resized = cv2.resize(person_crop, (int(person_crop.shape[1] * scale), int(person_crop.shape[0] * scale)))
        canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        y_offset = (target_size[0] - resized.shape[0]) // 2
        x_offset = (target_size[1] - resized.shape[1]) // 2
        canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
        return canvas
    except Exception as e:
        logging.error(f"Error processing crop: {e}")
        return None


def process_video_improved(video_path, output_dir):
    global cap
    import time
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    tracker = PersonTracker(max_disappeared=30, min_iou=0.5)
    classifier = CLIPClassifier(activity_labels=activity_labels)
    activity_distribution = defaultdict(int)
    sentiment_distribution = defaultdict(int)
    total_frames = 0
    analyzed_frames = 0
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Erro ao abrir o vídeo")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        output_video_path = os.path.join(output_dir, "processed_video.mp4")

        # Definir o codec e criar o objeto VideoWriter
        fourcc = int(cv2.CAP_PROP_FOURCC)
        analyzed_fps = video_fps / SKIP_FRAMES if SKIP_FRAMES > 0 else video_fps
        out_video = cv2.VideoWriter(output_video_path, fourcc, analyzed_fps, frame_size)

        # Criar diretórios para atividades
        activities_dir = os.path.join(output_dir, "activities")
        os.makedirs(activities_dir, exist_ok=True)
        for activity in activity_labels + ["Not Person", "Unknown"]:
            activity_folder = os.path.join(activities_dir, activity.replace(" ", "_"))
            os.makedirs(activity_folder, exist_ok=True)

        # Criar diretório para Tracker
        tracker_dir = os.path.join(output_dir, "Tracker")
        os.makedirs(tracker_dir, exist_ok=True)

        # Criar diretório para Sentiment
        sentiment_dir = os.path.join(output_dir, "Sentiment")
        os.makedirs(sentiment_dir, exist_ok=True)

        yolo_model = YOLO('yolo11x.pt')
        yolo_model = YOLO("yolo11n-seg.pt")  # Carregar um modelo oficial de segmentação
        report_file = os.path.join(output_dir, "frame_reports.jsonl")
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with ThreadPoolExecutor(max_workers=2) as executor, open(report_file, 'w') as report_f, tqdm(
                total=total_frame_count, desc="Processando vídeo") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                pbar.update(1)
                if frame_count % SKIP_FRAMES != 0:
                    continue
                frame = optimize_frame(frame)
                if frame is None:
                    continue

                results = yolo_model(frame, conf=0.5, classes=[0])
                boxes = [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])) for r in
                         results for box in r.boxes]
                current_trackers = tracker.update(boxes)

                if not current_trackers:
                    # Nenhuma pessoa detectada, definir atividade como "Not Person"
                    top_activities = [("Not Person", 1.0)]
                    frame_data = {
                        "Top_Activities": top_activities
                    }
                    # Desenhar "Not Person" na parte inferior do frame
                    cv2.putText(frame, " Not Person", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)
                    activity_distribution["Not Person"] += 1

                    # Salvar o frame na pasta da atividade "Not_Person"
                    activity1_label = "Not_Person"
                    activity_folder = os.path.join(activities_dir, activity1_label)
                    activity_frame_path = os.path.join(activity_folder,
                                                       f"frame_{frame_count:04d}_Label_{activity1_label}.jpg")
                    cv2.imwrite(activity_frame_path, frame)

                    # Escrever o frame no vídeo de saída
                    out_video.write(frame)

                    # Escrever no arquivo de relatório
                    json.dump({"Frame": frame_count, "People": frame_data}, report_f)
                    report_f.write('\n')
                    continue  # Ir para o próximo frame
                else:
                    # Existem pessoas detectadas, proceder com a classificação de atividade
                    # Classificar o frame inteiro
                    top_activities = classifier.classify_frame(frame)
                    logging.info(f"Top activities for frame {frame_count}: {top_activities}")

                    # Desenhar apenas a Activity 1 no frame na parte inferior
                    if top_activities:
                        activity1, score1 = top_activities[0]
                        cv2.putText(frame, f" {activity1}", (10, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        activity_distribution[activity1] += 1
                    else:
                        activity1 = "Unknown"
                        score1 = 0.0
                        cv2.putText(frame, f" {activity1} ({score1:.2f})", (10, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        activity_distribution[activity1] += 1

                    # No relatório, manter Activity 1 e 2
                    frame_data = {
                        "Top_Activities": top_activities
                    }

                    # Salvar o frame na pasta da atividade correspondente
                    activity1_label = activity1.replace(" ", "_") if activity1 else "Unknown"
                    activity_folder = os.path.join(activities_dir, activity1_label)
                    activity_frame_path = os.path.join(activity_folder,
                                                       f"frame_{frame_count:04d}_Label_{activity1_label}.jpg")
                    cv2.imwrite(activity_frame_path, frame)

                analyzed_frames += 1
                processed_crops = {}
                futures = {
                    track_id: executor.submit(analyze_sentiment_safe,
                                              process_person_crop(frame, data['bbox'], TARGET_SIZE))
                    for track_id, data in current_trackers.items()
                }
                for track_id, data in current_trackers.items():
                    bbox = data['bbox']
                    processed_crop = process_person_crop(frame, bbox, TARGET_SIZE)
                    if processed_crop is not None:
                        classifier.update_frames(track_id, processed_crop)
                        # Criar pasta do track ID se não existir
                        track_folder = os.path.join(tracker_dir, f"track_{track_id}")
                        os.makedirs(track_folder, exist_ok=True)
                        # Salvar a imagem do bounding box
                        bbox_image_path = os.path.join(track_folder, f"frame_{frame_count:04d}_track_{track_id}.jpg")
                        cv2.imwrite(bbox_image_path, processed_crop)
                        # Armazenar o crop processado para uso posterior
                        processed_crops[track_id] = processed_crop

                        # Desenhar bounding box e ID no canto superior direito
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        id_text = f"ID:{track_id}"
                        (text_width, text_height), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_x = bbox[2] - text_width - 5  # 5 pixels de padding
                        text_y = bbox[1] + text_height + 5  # 5 pixels de padding
                        cv2.putText(frame, id_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                for track_id, future in futures.items():
                    sentiment = future.result()
                    if isinstance(sentiment, list):
                        sentiment = sentiment[0]  # Usar o primeiro sentimento, se necessário
                    frame_data[str(track_id)] = {
                        "Sentiment": sentiment
                    }
                    # Atualizar distribuição de sentimentos
                    sentiment_distribution[sentiment] += 1
                    # Salvar a imagem do bounding box na pasta do sentimento
                    sentiment_folder = os.path.join(sentiment_dir, sentiment)
                    os.makedirs(sentiment_folder, exist_ok=True)
                    processed_crop = processed_crops.get(track_id)
                    if processed_crop is not None:
                        sentiment_image_path = os.path.join(sentiment_folder,
                                                            f"frame_{frame_count:04d}_track_{track_id}.jpg")
                        cv2.imwrite(sentiment_image_path, processed_crop)

                # Escrever o frame no vídeo de saída
                out_video.write(frame)

                json.dump({"Frame": frame_count, "People": frame_data}, report_f)
                report_f.write('\n')
                classifier.cleanup_old_tracks(list(current_trackers.keys()))
                if frame_count % 100 == 0:
                    gc.collect()
        # Após o loop de processamento
        out_video.release()
        cap.release()
        total_time = time.time() - start_time
        analyzed_fps = analyzed_frames / total_time if total_time > 0 else 0
        summary = {
            "Total_Frames": frame_count,
            "Analyzed_Frames": analyzed_frames,
            "Unique_IDs": tracker.next_id,
            "Activity_Distribution": dict(activity_distribution),
            "Sentiment_Distribution": dict(sentiment_distribution),
            "Video_FPS": video_fps,
            "Analyzed_FPS": analyzed_fps,
            "Total_Processing_Time_sec": total_time,
            "Average_Time_Per_Frame_sec": total_time / analyzed_frames if analyzed_frames > 0 else None
        }
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        if out_video.isOpened():
            out_video.release()
        torch.cuda.empty_cache()
        gc.collect()


# Execução do processo
if __name__ == "__main__":
    try:
        video_path = "data/tech_chall4.mp4"  # Atualize com o caminho para o seu vídeo
        output_dir = "output/output_test1"  # Atualize com o diretório de saída desejado
        process_video_improved(video_path, output_dir)
        logging.info("Processing completed successfully!")
    except Exception as e:
        logging.error(f"Error during execution: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
