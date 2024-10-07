# Importing Libraries

import os
import csv
import random
from collections import Counter

import numpy as np # type: ignore
import torch  # type: ignore 
import tensorflow as tf # type: ignore
from PIL import Image # type: ignore

from django.conf import settings
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

from joblib import load # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

from transformers import ( # type: ignore
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
)

from .serializers import PredictionSerializer


# Initializing Paths

QUESTION_GENERATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'T5_models', 'T5QuestionGenerationModel')
ANSWER_GENERATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'T5_models', 'T5AnswerGenerationModel')
DISTRACTOR_GENERATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'T5_models', 'T5DistractorGenerationModel')
PASSAGE_PATH = os.path.join(settings.BASE_DIR, 'T5_models', 'narrativeqa_summaries.csv')
RESNET50V2_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ResNet50V2_models_and_labelencoder', 'resnet50v2_model.keras')
AROUSAL_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ResNet50V2_models_and_labelencoder','label_encoder_arousal.pkl')
DOMINANCE_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ResNet50V2_models_and_labelencoder', 'label_encoder_dominance.pkl')




#######################################################################################

#################### NATURAL LANGUAGE MODELS (T5) #########################

#######################################################################################


def classify_question_type(question: str) -> str:
    """
    Classify the type of question as literal, evaluative, or inferential.
    
    Parameters:
        question (str): The question to classify.
        
    Returns:
        str: The type of the question ('literal', 'evaluative', or 'inferential').
    """
    # Define keywords or patterns for each question type
    literal_keywords = [
    'what', 'when', 'where', 'who', 'how many', 'how much', 
    'which', 'name', 'list', 'identify', 'define', 'describe', 
    'state', 'mention'
    ]

    evaluative_keywords = [
    'evaluate', 'justify', 'explain why', 'assess', 'critique', 
    'discuss', 'judge', 'opinion', 'argue', 'agree or disagree', 
    'defend', 'support your answer', 'weigh the pros and cons', 
    'compare', 'contrast'
    ]

    inferential_keywords = [
    'why', 'how', 'what if', 'predict', 'suggest', 'imply', 
    'conclude', 'infer', 'reason', 'what might', 'what could', 
    'what would happen if', 'speculate', 'deduce', 'interpret', 
    'hypothesize', 'assume'
    ]


    question_lower = question.lower()
    
    # Check for literal question keywords
    if any(keyword in question_lower for keyword in literal_keywords):
        return 'literal'
    
    # Check for evaluative question keywords
    if any(keyword in question_lower for keyword in evaluative_keywords):
        return 'evaluative'
    
    # Check for inferential question keywords
    if any(keyword in question_lower for keyword in inferential_keywords):
        return 'inferential'
    
    # Default to 'unknown' if no pattern matches
    return 'unknown'

# Global instances for models
QG = None
AG = None
DG = None

class QuestionGeneration:
    def __init__(self):
        print("Loading model from:", QUESTION_GENERATION_MODEL_PATH)
        self.model = T5ForConditionalGeneration.from_pretrained(QUESTION_GENERATION_MODEL_PATH)
        self.tokenizer = T5Tokenizer.from_pretrained(QUESTION_GENERATION_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Question Generation model loaded successfully.")

    def generate(self, answer: str, passage: str):
        input_text = '<answer> %s <context> %s ' % (answer, passage)
        encoding = self.tokenizer.encode_plus(
            input_text,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        question = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return question

class AnswerGeneration:
    def __init__(self):
        print("Loading model from:", ANSWER_GENERATION_MODEL_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(ANSWER_GENERATION_MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(ANSWER_GENERATION_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Answer Generation model loaded successfully")

    def generate(self, question: str, passage: str):
        input_text = f"{self.tokenizer.cls_token}{question}"
        encoding = self.tokenizer(
            input_text,
            passage,
            return_tensors="pt")
        output = self.model(
            encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

        all_tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
        answer_tokens = all_tokens[torch.argmax(output["start_logits"]):torch.argmax(output["end_logits"]) + 1]
        answer = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(answer_tokens))

        return answer

class DistractorGeneration:
    def __init__(self):
        print("Loading model from:", DISTRACTOR_GENERATION_MODEL_PATH)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(DISTRACTOR_GENERATION_MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(DISTRACTOR_GENERATION_MODEL_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Distractor Generation model loaded successfully")

    def generate(self, question: str, answer: str, passage: str):
        input_text = " ".join([question, self.tokenizer.sep_token, answer, self.tokenizer.sep_token, passage])
        encoding = self.tokenizer(
            input_text,
            return_tensors="pt")
        outputs = self.model.generate(**encoding, max_new_tokens=128)
        distractors = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        distractors = distractors.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
        distractors = [y.strip() for y in distractors.split(self.tokenizer.sep_token)]

        return distractors

# Keyword Extraction Function
def get_keywords(passage, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([passage])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:num_keywords]]
    return top_keywords

def get_random_passage(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        passages = list(reader)
        header = passages.pop(0)
        random_passage = random.choice(passages)
    return random_passage


class PredictionAPIView(APIView):
    def post(self, request):
        global QG, AG, DG
        
        # Load models if they are not already loaded
        if QG is None:
            QG = QuestionGeneration()
        if AG is None:
            AG = AnswerGeneration()
        if DG is None:
            DG = DistractorGeneration()

        # Extract the passage from the request data
        passage_input = request.data.get('passage', None)

        print(f"PASSAGE : {passage_input}")
        
        if passage_input is None:
            return Response({"error": "No passage provided."}, status=status.HTTP_400_BAD_REQUEST)
        

        # Optionally, you can also clean or validate the input passage if needed

        # Extract up to 5 keywords from the provided passage
        keywords = get_keywords(passage_input, num_keywords=5)

        # Initialize a dict to store questions-choices-answer objects
        questions_and_answers_dict = {}

        if keywords:
            # Loop over the extracted keywords to generate multiple questions and answers
            for idx, keyword in enumerate(keywords):
                generated_question = QG.generate(keyword, passage_input)
                generated_answer = AG.generate(generated_question, passage_input)
                generated_distractors = DG.generate(generated_question, generated_answer, passage_input)

                # Classify the question type
                question_type = classify_question_type(generated_question)

                # Store the result in the required format using unique keys like "question_1", "question_2", etc.
                questions_and_answers_dict[f"question_{idx+1}"] = {
                    "question": generated_question,
                    "choices": generated_distractors,
                    "answer": generated_answer,
                    "question_type": question_type  # Use the classified question type
                }

        else:
            # If no keywords found, return a default message
            questions_and_answers_dict["question_1"] = {
                "question": "No question generated",
                "choices": [],
                "answer": "N/A",
                "question_type": "literal"
            }

        # Prepare the final response with the passage and questions-choices-answer dict
        response_data = {
            'passage': passage_input,
            'questions-choices-answer': questions_and_answers_dict
        }

        return Response(response_data, status=status.HTTP_200_OK)

def test(request):
    global QG, AG, DG

    # Load models if they are not already loaded
    if QG is None:
        QG = QuestionGeneration()
    if AG is None:
        AG = AnswerGeneration()
    if DG is None:
        DG = DistractorGeneration()

    # Extract a random passage and its string representation
    random_passage = get_random_passage(PASSAGE_PATH)
    random_passage_str = ' '.join(map(str, random_passage))

    # Extract up to 5 keywords from the passage
    keywords = get_keywords(random_passage_str, num_keywords=5)

    questions_and_answers = []

    if keywords:
        # Loop over each keyword to generate question, answer, and distractors
        for keyword in keywords:
            generated_question = QG.generate(keyword, random_passage_str)
            generated_answer = AG.generate(generated_question, random_passage_str)
            generated_distractors = DG.generate(generated_question, generated_answer, random_passage_str)

            questions_and_answers.append({
                'keyword': keyword,
                'question': generated_question,
                'answer': generated_answer,
                'choices': generated_distractors
            })
    else:
        questions_and_answers.append({
            'keyword': "No keywords found",
            'question': "No question generated",
            'answer': "N/A",
            'choices': []
        })

    print(f"Generated {len(questions_and_answers)} Questions")

    # Prepare context for rendering
    context = {
        'passage': random_passage_str,
        'questions-choices-answer': questions_and_answers
    }

    return render(request, 'index.html', context)


#######################################################################################

#################### COMPUTER VISION MODEL (ResNet50V2.keras) #########################

#######################################################################################

class ResNet50V2Model:
    def __init__(self):
        # Load the model
        self.model = self.load_model(RESNET50V2_MODEL_PATH)

        # Load encoders for arousal and dominance
        self.arousal_encoder = self.load_encoder(AROUSAL_ENCODER_PATH, "Arousal")
        self.dominance_encoder = self.load_encoder(DOMINANCE_ENCODER_PATH, "Dominance")

    def load_model(self, model_path):
        """Load the ResNet50V2 model from the specified path."""
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            return tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def load_encoder(self, encoder_path, encoder_type):
        """Load a OneHotEncoder from the specified path."""
        if os.path.exists(encoder_path):
            print(f"Loading OneHotEncoder for {encoder_type} from {encoder_path}...")
            return load(encoder_path)
        else:
            raise FileNotFoundError(f"{encoder_type} encoder file not found at {encoder_path}")

    def preprocess_image(self, image):
        """Preprocess the uploaded image to be compatible with the model."""
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        img = image.resize((256, 256))  # Resize to the expected input size
        img_array = np.array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image
        return img_array

    def predict(self, images):
        """Make predictions on the processed images."""
        results = []
        continuous_sums = None

        # Preprocess all images at once
        processed_images = np.array([self.preprocess_image(image) for image in images])
        processed_images = np.vstack(processed_images)  # Stack to create a batch

        # Run the prediction using the model
        predictions = self.model.predict(processed_images)

        # Debug: Print the predictions to understand their structure
        print("Predictions shape:", {k: v.shape for k, v in predictions.items()})

        # Loop through the predictions for each image
        for idx in range(len(images)):
            try:
                # Extract predictions for the current image
                arousal_pred = predictions['arousal_output'][idx]  # Access the current element
                dominance_pred = predictions['dominance_output'][idx]  # Access the current element
                continuous_pred = predictions['continuous_output'][idx]  # Access the current element

                # Store results
                results.append({
                    "arousal": arousal_pred,
                    "dominance": dominance_pred,
                    "continuous": continuous_pred
                })

                # Debug: Print individual results
                print(f"Image {idx + 1} results: {results[-1]}")

                # Calculate sums for continuous outputs
                if continuous_sums is None:
                    continuous_sums = continuous_pred
                else:
                    continuous_sums += continuous_pred

            except Exception as e:
                print(f"Error processing image {idx + 1}: {e}")

        # Ensure results are available
        if not results:
            raise ValueError("No valid predictions were made. Please check the input images.")

        # Calculate final results
        final_arousal = Counter([np.argmax(res['arousal']) for res in results]).most_common(1)[0][0]
        final_dominance = Counter([np.argmax(res['dominance']) for res in results]).most_common(1)[0][0]

        continuous_averages = continuous_sums / len(images)

        # Decode the final labels
        arousal_label = self.arousal_encoder.inverse_transform(
            np.eye(self.arousal_encoder.categories_[0].shape[0])[final_arousal].reshape(1, -1)
        )
        dominance_label = self.dominance_encoder.inverse_transform(
            np.eye(self.dominance_encoder.categories_[0].shape[0])[final_dominance].reshape(1, -1)
        )

        # Define class names for continuous output
        class_names = ['effort', 'frustration', 'mental_demand', 'performance', 'physical_demand']
        continuous_results = {name: continuous_averages[i] for i, name in enumerate(class_names)}

        # Return formatted predictions
        return {
            "arousal": arousal_label[0][0],
            "dominance": dominance_label[0][0],
            "continuous": continuous_results
        }



def calculate_prediction_results(results):
    """
    Calculate the final prediction results from model outputs.

    Args:
        results (list): A list of dictionaries containing arousal, dominance, and continuous results.

    Returns:
        dict: A dictionary containing the most common arousal, most common dominance, and average continuous results.
    """
    # Extract arousal and dominance results as strings
    arousal_values = [result['arousal'] for result in results]
    dominance_values = [result['dominance'] for result in results]
    
    # Extract continuous results
    continuous_results = {}
    for result in results:
        for key, value in result['continuous'].items():
            if key not in continuous_results:
                continuous_results[key] = []
            continuous_results[key].append(value)

    # Calculate most common arousal and dominance
    most_common_arousal = Counter(arousal_values).most_common(1)[0][0]
    most_common_dominance = Counter(dominance_values).most_common(1)[0][0]

    # Calculate averages for continuous results
    average_continuous = {key: np.mean(values) for key, values in continuous_results.items()}

    # Return formatted results
    return {
        "arousal": most_common_arousal,
        "dominance": most_common_dominance,
        "average_continuous": average_continuous
    }


class ResNet50V2APIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        # Handle the image upload from the form
        image_files = request.FILES.getlist('images')  # Expecting multiple images

        if not image_files:
            return Response({"error": "No images were uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        model = ResNet50V2Model()
        results = []

        try:
            for image_file in image_files:
                image = Image.open(image_file)
                result = model.predict([image])  # Wrap in list for single prediction
                results.append({
                    'filename': image_file.name,
                    'arousal': result['arousal'],
                    'dominance': result['dominance'],
                    'continuous': result['continuous'],
                })

            overall_results = calculate_prediction_results(results)

            return Response({
                "results": results,
                "overall_results": overall_results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




# class ResNet50V2APIView(APIView):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # Initialize the model class
#         self.model = ResNet50V2Model()

#     def post(self, request):
#         image_file = request.FILES.get('image')
#         if not image_file:
#             return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             # Convert the uploaded image into a format suitable for PIL
#             image = Image.open(image_file)

#             # Make predictions using the ResNet50V2Model class
#             result = self.model.predict(image)

#             return Response({"result": result}, status=status.HTTP_200_OK)

#         except Exception as e:
#             # Handle any exceptions during processing
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)





def test(request):
    context = {}

    if request.method == 'POST':
        # Handle the image upload from the form
        image_files = request.FILES.getlist('image')  # Get all uploaded images

        print(f"Number of images uploaded: {len(image_files)}")
        
        if image_files:
            try:
                # Initialize the ResNet50V2Model
                model = ResNet50V2Model()

                # Prepare images for prediction
                images = [Image.open(image_file) for image_file in image_files]

                # Get predictions for the uploaded images
                results = []
                for idx, image in enumerate(images):
                    # Get the prediction for each image
                    result = model.predict([image])  # Wrap in list for single prediction
                    # Add filename and prediction results to the list
                    results.append({
                        'filename': image_files[idx].name,  # Get the original filename
                        'arousal': result['arousal'],
                        'dominance': result['dominance'],
                        'continuous': result['continuous'],
                    })

                # Calculate overall prediction results
                overall_results = calculate_prediction_results(results)

                # Add the results and overall results to context to display on the page
                context['results'] = results
                context['overall_results'] = overall_results  # Include overall results for rendering

            except Exception as e:
                # If there's an error, add it to the context
                context['error'] = str(e)

        else:
            context['error'] = "No images were uploaded."
    
    return render(request, 'index.html', context)