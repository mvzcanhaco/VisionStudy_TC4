# activity_classifier.py

import torch
import open_clip
from PIL import Image, ImageDraw, ImageFont
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np


class ActivityClassifier:
    """Classe para classificação de atividades em frames usando OpenCLIP."""

    def __init__(
        self,
        execution_number: int,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai"
    ):
        """
        Inicializa o classificador de atividades.

        Args:
            execution_number: Número da execução atual
            model_name: Nome do modelo OpenCLIP a ser usado
            pretrained: Dataset de pré-treinamento do modelo
        """
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.device = device
        self.execution_number = execution_number
        self.output_dir = Path(f"Outputs/Exec_{execution_number}/Activity_Frames")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuração do modelo OpenCLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Configuração de logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configurar fonte para anotação
        try:
            self.font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            self.font = ImageFont.load_default()

        # Lista de atividades
        self.activity_labels = [
            "reading",
            "group_activity",
            "computer_work",
            "medical",
            "phone_usage",
            "resting",
            "dancing",
            "standing",
            "making_funny",
            "anomaly",
        ]

        # Dicionário com descrições detalhadas para atividades específicas
        self.ACTIVITY_DESCRIPTIONS = {
            "reading": [
                "a person reading a book",
                "people sitting and reading quietly",
                "individual reading a document attentively",
                "someone reading on a tablet or phone",
                "group of people reading together in silence",
                "a person focused on reading",
                "people gathered and reading books",
            ],
            "group_activity": [
                "a group of people working together",
                "people collaborating in an office",
                "team members discussing a project",
                "colleagues brainstorming ideas",
                "group of individuals in a meeting",
                "people gathered around a table discussing",
            ],
            "computer_work": [
                "a person using a computer",
                "someone typing on a keyboard",
                "individual working on a laptop",
                "person focused on computer screen",
                "using a desktop computer",
                "people working with computers in an office",
            ],
            "medical": [
                "a doctor and patient in a medical consultation",
                "doctor talking to patient",
                "medical professionals in a consultation",
                "doctor examining a patient",
                "patient discussing with doctor",
                "medical procedure in progress",
            ],
            "phone_usage": [
                "person using a cell phone",
                "someone texting on smartphone",
                "individual looking at mobile phone",
                "person browsing on phone",
                "using a smartphone device",
                "people checking their phones",
            ],
            "resting": [
                "a person lying down and sleeping",
                "individual resting on a bed",
                "someone sleeping peacefully",
                "person lying down with eyes closed",
                "sleeping person on a couch",
                "people relaxing in an office",
            ],
            "dancing": [
                "a person dancing",
                "someone doing a dance performance",
                "ballet dancer in motion",
                "a person ballet dancing",
                "a person dancing energetically",
                "someone performing a dance routine",
                "individual dancing at a party",
                "person showing dance moves",
                "group of people dancing together",
                "a dancer in motion",
            ],
            "standing": [
                "a person standing and looking directly at the camera",
                "individual posing while standing",
                "person standing still facing the camera",
                "someone standing upright looking forward",
                "person standing and making eye contact with the camera",
            ],
            "making_funny": [
                "a person making funny faces",
                "someone acting silly and making faces",
                "individual making comical expressions",
                "person pulling funny faces",
                "someone making goofy faces",
                "person doing funny expressions",
                "individual making silly faces",
                "someone being silly and playful",
                "person acting funny and making faces",
                "someone making humorous expressions",
                "individual being playful and silly"
            ],
            "anomaly": [
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

        # Gerar prompts para cada atividade
        self.activities = {
            activity: self.generate_clip_prompts(activity)
            for activity in self.activity_labels
        }

        # Pré-computar embeddings dos textos
        self.text_embeddings = self._precompute_text_embeddings()
        self.logger.info(
            f"Initialized ActivityClassifier with model {model_name} on {self.device}"
        )

    def generate_clip_prompts(self, action: str) -> List[str]:
        """
        Gera prompts formatados para o CLIP baseado na ação.
        """
        base_prompts = self.ACTIVITY_DESCRIPTIONS.get(
            action, [f"a person {action.replace('_', ' ')}"]
        )
        enhanced_prompts = []

        for prompt in base_prompts:
            # Adicionar variações de contexto
            enhanced_prompts.extend(
                [
                    prompt,
                    f"a photo of {prompt}",
                    f"an image of {prompt}",
                    f"a picture of {prompt}",
                    f"a snapshot of {prompt}",
                    f"a video frame showing {prompt}",
                    f"a depiction of {prompt}",
                    f"a representation of {prompt}",
                    f"an illustration of {prompt}",
                ]
            )

        return enhanced_prompts

    def _precompute_text_embeddings(self) -> torch.Tensor:
        """Pré-computa os embeddings de texto para todas as atividades."""
        all_prompts = []
        for prompts in self.activities.values():
            all_prompts.extend(prompts)

        with torch.no_grad():
            text = self.tokenizer(all_prompts).to(self.device)
            text_features = self.model.encode_text(text)
            normalized_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )
            return normalized_features

    def _add_activity_label(
        self, image: Image.Image, activity: str, confidence: float
    ) -> Image.Image:
        """
        Adiciona o rótulo da atividade no canto inferior esquerdo da imagem.

        Args:
            image: Imagem a ser anotada
            activity: Nome da atividade detectada
            confidence: Confiança da classificação

        Returns:
            Imagem anotada com o rótulo da atividade
        """
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        text = f"{activity} ({confidence:.2f})"
        margin = 10
        text_bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = (margin, image.height - text_height - margin)

        # Adicionar fundo semi-transparente
        bg_bbox = [
            position[0] - 5,
            position[1] - 5,
            position[0] + text_width + 5,
            position[1] + text_height + 5,
        ]
        draw.rectangle(bg_bbox, fill=(0, 0, 0, 128))

        # Desenhar texto
        draw.text(position, text, fill=(255, 255, 255), font=self.font)

        return annotated_image

    def process_frame(
        self, frame: Image.Image, frame_name: str
    ) -> Tuple[str, float, Image.Image]:
        """
        Processa um único frame e retorna sua classificação.

        Args:
            frame: Imagem do frame a ser processado
            frame_name: Nome do arquivo do frame

        Returns:
            Tupla contendo (atividade detectada, confiança, imagem anotada)
        """
        try:
            # Preprocessar frame
            processed_frame = self.preprocess(frame).unsqueeze(0).to(self.device)

            # Obter embeddings da imagem
            with torch.no_grad():
                image_features = self.model.encode_image(processed_frame)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                # Calcular similaridade com todos os prompts
                similarity = (
                    100.0 * image_features @ self.text_embeddings.T
                ).softmax(dim=-1)

                # Calcular média de similaridade por atividade
                scores = {}
                start_idx = 0
                for activity, prompts in self.activities.items():
                    end_idx = start_idx + len(prompts)
                    activity_similarity = similarity[0, start_idx:end_idx]
                    # Usar o score máximo em vez da média
                    activity_score = activity_similarity.max().item()
                    scores[activity] = activity_score
                    start_idx = end_idx

                # Encontrar atividade com maior score
                best_activity, confidence = max(
                    scores.items(), key=lambda x: x[1]
                )

                # Adicionar anotação na imagem
                annotated_image = self._add_activity_label(
                    frame, best_activity, confidence
                )

                # Salvar imagem processada
                output_path = self.output_dir / best_activity
                output_path.mkdir(exist_ok=True)
                annotated_image.save(output_path / frame_name, quality=95)

                self.logger.debug(
                    f"Processed frame {frame_name}: {best_activity} ({confidence:.2f})"
                )
                return best_activity, confidence, annotated_image

        except Exception as e:
            self.logger.error(f"Error processing frame {frame_name}: {str(e)}")
            return "unknown", 0.0, frame
