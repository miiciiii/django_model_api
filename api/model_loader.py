import os
from collections import Counter

import numpy as np  # type: ignore
import torch  # type: ignore 
import tensorflow as tf  # type: ignore

from django.conf import settings
from joblib import load  # type: ignore

from transformers import (  # type: ignore
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
)

# Define model paths
QUESTION_GENERATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'T5_models', 'T5QuestionGenerationModel')
ANSWER_GENERATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'T5_models', 'T5AnswerGenerationModel')
DISTRACTOR_GENERATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'T5_models', 'T5DistractorGenerationModel')
RESNET50V2_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ResNet50V2_models_and_labelencoder', 'ResNet50V2t3.keras')
AROUSAL_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ResNet50V2_models_and_labelencoder', 'Arousal_label_encoder.joblib')
DOMINANCE_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ResNet50V2_models_and_labelencoder', 'Dominance_label_encoder.joblib')
MIN_MAX_SCALER_PATH = os.path.join(settings.BASE_DIR, 'ResNet50V2_models_and_labelencoder', 'min_max_scaler.joblib')

# Initialize model instances

QGMODEL = T5ForConditionalGeneration.from_pretrained(QUESTION_GENERATION_MODEL_PATH)
AGMODEL = AutoModelForQuestionAnswering.from_pretrained(ANSWER_GENERATION_MODEL_PATH)
DGMODEL = AutoModelForSeq2SeqLM.from_pretrained(DISTRACTOR_GENERATION_MODEL_PATH)

QG = None
AG = None
DG = None
ResNet50 = None


class QuestionGeneration:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QuestionGeneration, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print("Loading model from:", QUESTION_GENERATION_MODEL_PATH)
        self.model = QGMODEL
        self.tokenizer = T5Tokenizer.from_pretrained(QUESTION_GENERATION_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Question Generation model loaded successfully.")
        self._initialized = True

    def generate(self, answer: str, passage: str):
        input_text = '<answer> %s <context> %s ' % (answer, passage)
        encoding = self.tokenizer.encode_plus(input_text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return question


class AnswerGeneration:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AnswerGeneration, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print("Loading model from:", ANSWER_GENERATION_MODEL_PATH)
        self.model = AGMODEL
        self.tokenizer = AutoTokenizer.from_pretrained(ANSWER_GENERATION_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Answer Generation model loaded successfully")
        self._initialized = True

    def generate(self, question: str, passage: str):
        input_text = f"{self.tokenizer.cls_token}{question}"
        encoding = self.tokenizer(input_text, passage, return_tensors="pt", truncation=True, max_length=512)
        output = self.model(encoding["input_ids"], attention_mask=encoding["attention_mask"])

        all_tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
        answer_tokens = all_tokens[torch.argmax(output["start_logits"]):torch.argmax(output["end_logits"]) + 1]
        answer = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(answer_tokens))

        return answer


class DistractorGeneration:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DistractorGeneration, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print("Loading model from:", DISTRACTOR_GENERATION_MODEL_PATH)
        self.model = DGMODEL
        self.tokenizer = AutoTokenizer.from_pretrained(DISTRACTOR_GENERATION_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Distractor Generation model loaded successfully")
        self._initialized = True

    def generate(self, question: str, answer: str, passage: str):
        input_text = " ".join([question, self.tokenizer.sep_token, answer, self.tokenizer.sep_token, passage])
        encoding = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**encoding, max_new_tokens=128)
        distractors = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        distractors = distractors.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
        distractors = [y.strip() for y in distractors.split(self.tokenizer.sep_token)]
        return distractors


# class ResNet50V2Model:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super(ResNet50V2Model, cls).__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance

#     def __init__(self):
#         if self._initialized:
#             return
#         self.model = self.load_model(RESNET50V2_MODEL_PATH)
#         self.arousal_encoder = self.load_encoder(AROUSAL_ENCODER_PATH, "Arousal")
#         self.dominance_encoder = self.load_encoder(DOMINANCE_ENCODER_PATH, "Dominance")
#         self._initialized = True

#     def load_model(self, model_path):
#         if os.path.exists(model_path):
#             print(f"Loading model from {model_path}...")
#             return tf.keras.models.load_model(model_path)
#         else:
#             raise FileNotFoundError(f"Model file not found at {model_path}")

#     def load_encoder(self, encoder_path, encoder_type):
#         if os.path.exists(encoder_path):
#             print(f"Loading OneHotEncoder for {encoder_type} from {encoder_path}...")
#             return load(encoder_path)
#         else:
#             raise FileNotFoundError(f"{encoder_type} encoder file not found at {encoder_path}")

#     def preprocess_image(self, image):
#         if image.mode == 'RGBA':
#             image = image.convert('RGB')
#         img = image.resize((256, 256))
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0
#         return img_array

#     def predict(self, images):
#         results = []
#         continuous_sums = None
#         processed_images = np.array([self.preprocess_image(image) for image in images])
#         processed_images = np.vstack(processed_images)
#         predictions = self.model.predict(processed_images)
#         print("Predictions shape:", {k: v.shape for k, v in predictions.items()})
#         for idx in range(len(images)):
#             try:
#                 arousal_pred = predictions['arousal_output'][idx]
#                 dominance_pred = predictions['dominance_output'][idx]
#                 continuous_pred = predictions['continuous_output'][idx]
#                 results.append({
#                     "arousal": arousal_pred,
#                     "dominance": dominance_pred,
#                     "continuous": continuous_pred
#                 })
#                 if continuous_sums is None:
#                     continuous_sums = continuous_pred
#                 else:
#                     continuous_sums += continuous_pred
#             except Exception as e:
#                 print(f"Error processing image {idx + 1}: {e}")

#         if not results:
#             raise ValueError("No valid predictions were made. Please check the input images.")

#         final_arousal = Counter([np.argmax(res['arousal']) for res in results]).most_common(1)[0][0]
#         final_dominance = Counter([np.argmax(res['dominance']) for res in results]).most_common(1)[0][0]
#         continuous_averages = continuous_sums / len(images)

#         arousal_label = self.arousal_encoder.inverse_transform(
#             np.eye(self.arousal_encoder.categories_[0].shape[0])[final_arousal].reshape(1, -1)
#         )
#         dominance_label = self.dominance_encoder.inverse_transform(
#             np.eye(self.dominance_encoder.categories_[0].shape[0])[final_dominance].reshape(1, -1)
#         )

#         class_names = ['effort', 'frustration', 'mental_demand', 'performance', 'physical_demand']
#         continuous_results = {name: continuous_averages[i] for i, name in enumerate(class_names)}

#         return {
#             "arousal": arousal_label[0][0],
#             "dominance": dominance_label[0][0],
#             "continuous": continuous_results
#         }


class ResNet50V2Model:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ResNet50V2Model, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = self.load_model(RESNET50V2_MODEL_PATH)
        self.arousal_encoder = self.load_encoder(AROUSAL_ENCODER_PATH, "Arousal")
        self.dominance_encoder = self.load_encoder(DOMINANCE_ENCODER_PATH, "Dominance")
        self.min_max_scaler = self.load_scaler(MIN_MAX_SCALER_PATH)
        self._initialized = True

    def load_model(self, model_path):
        """Load the pre-trained model."""
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            return tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def load_encoder(self, encoder_path, encoder_type):
        """Load the LabelEncoder for categorical outputs."""
        if os.path.exists(encoder_path):
            print(f"Loading LabelEncoder for {encoder_type} from {encoder_path}...")
            return load(encoder_path)
        else:
            raise FileNotFoundError(f"{encoder_type} encoder file not found at {encoder_path}")

    def load_scaler(self, scaler_path):
        """Load the MinMaxScaler for continuous outputs."""
        if os.path.exists(scaler_path):
            print(f"Loading MinMaxScaler from {scaler_path}...")
            return load(scaler_path)
        else:
            raise FileNotFoundError(f"MinMaxScaler file not found at {scaler_path}")

    def preprocess_image(self, image):
        """Preprocess a single image for model inference."""
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        img = image.resize((224, 224))  # Assuming model input size is 224X224
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        return img_array

    def predict(self, images):
        """
        Predict using the model and return outputs for all classes.

        Args:
            images (list): List of PIL.Image objects.

        Returns:
            list: A list of dictionaries with predictions for all outputs.
        """
        # Preprocess images
        processed_images = np.array([self.preprocess_image(image) for image in images])
        processed_images = np.vstack(processed_images)

        # Get predictions from the model
        predictions = self.model.predict(processed_images)
        print("Predictions received for all outputs.")

        # Separate outputs
        arousal_preds = predictions[0]  # Arousal (categorical)
        dominance_preds = predictions[1]  # Dominance (categorical)
        frustration_preds = predictions[2]  # Frustration (continuous)
        mental_demand_preds = predictions[3]  # Mental Demand (continuous)
        performance_preds = predictions[4]  # Performance (continuous)
        physical_demand_preds = predictions[5]  # Physical Demand (continuous)
        effort_preds = predictions[6]  # Effort (continuous)

        results = []
        for idx in range(len(images)):
            try:
                # Decode categorical predictions using LabelEncoder
                arousal_label = self.arousal_encoder.inverse_transform([np.argmax(arousal_preds[idx])])[0]
                dominance_label = self.dominance_encoder.inverse_transform([np.argmax(dominance_preds[idx])])[0]

                # Scale continuous predictions back to the original range
                continuous_values = np.array([ 
                    frustration_preds[idx][0], 
                    mental_demand_preds[idx][0], 
                    performance_preds[idx][0], 
                    physical_demand_preds[idx][0], 
                    effort_preds[idx][0]
                ]).reshape(1, -1)
                scaled_values = self.min_max_scaler.inverse_transform(continuous_values)[0]

                # Ensure continuous values are positive (take absolute values if negative)
                scaled_values = np.abs(scaled_values)

                # Cap continuous values at 21 if they exceed it
                scaled_values = np.minimum(scaled_values, 21)

                # Append predictions to results
                results.append({
                    "arousal": arousal_label,
                    "dominance": dominance_label,
                    "frustration": float(scaled_values[0]),
                    "mental_demand": float(scaled_values[1]),
                    "performance": float(scaled_values[2]),
                    "physical_demand": float(scaled_values[3]),
                    "effort": float(scaled_values[4]),
                })

            except Exception as e:
                print(f"Error processing image {idx + 1}: {e}")

        if not results:
            raise ValueError("No valid predictions were made. Please check the input images.")

        return results



def initialize_models():
    global QG, AG, DG, ResNet50
    if QG is None:
        QG = QuestionGeneration()
    if AG is None:
        AG = AnswerGeneration()
    if DG is None:
        DG = DistractorGeneration()
    if ResNet50 is None:
        ResNet50 = ResNet50V2Model()
    return {
        'QG': QG,
        'AG': AG,
        'DG': DG,
        'ResNet50': ResNet50
    }
