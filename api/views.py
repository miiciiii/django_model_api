# Importing Libraries

from PIL import Image # type: ignore 

from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore


from .model_loader import QG, AG, DG, ResNet50, initialize_models
from . import utils as utils

initialize_models()

if QG is None or AG is None or DG is None or ResNet50 is None:
    raise RuntimeError("RUNTIME ERROR : MODELS ARE NOT INITIALIZED")


#######################################################################################

#################### NATURAL LANGUAGE MODELS (T5) #########################

#######################################################################################


class PredictionAPIView(APIView):
    def post(self, request):
        # Extract the passage from the request data
        passage_input = request.data.get('passage', None)

        print(f"PASSAGE : {passage_input}")
        
        if passage_input is None:
            return Response({"error": "No passage provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Optionally, you can also clean or validate the input passage if needed

        # Extract up to 5 keywords from the provided passage
        keywords = utils.get_keywords(passage_input, num_keywords=5)

        # Initialize a dict to store questions-choices-answer objects
        questions_and_answers_dict = {}

        if keywords:
            # Loop over the extracted keywords to generate multiple questions and answers
            for idx, keyword in enumerate(keywords):
                generated_question = QG.generate(keyword, passage_input)
                generated_answer = AG.generate(generated_question, passage_input)
                generated_distractors = DG.generate(generated_question, generated_answer, passage_input)

                # Classify the question type
                question_type = utils.classify_question_type(generated_question)

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
    

def testt5pred(request):
    random_passage_str = ''
    questions_and_answers = []

    if request.method == 'POST':
        # Get the passage from the form submission
        random_passage_str = request.POST.get('passage', '')

        # Extract up to 5 keywords from the passage after it has been received
        keywords = utils.get_keywords(random_passage_str, num_keywords=5)

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
        'questions_choices_answer': questions_and_answers
    }

    return render(request, 'testt5.html', context)



#######################################################################################

#################### COMPUTER VISION MODEL (ResNet50V2.keras) #########################

#######################################################################################


class ResNet50V2APIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        # Handle the image upload from the form
        image_files = request.FILES.getlist('images')  # Expecting multiple images

        if not image_files:
            return Response({"error": "No images were uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        try:
            for image_file in image_files:
                image = Image.open(image_file)
                result = ResNet50.predict([image])  # Use the global instance
                results.append({
                    'filename': image_file.name,
                    'arousal': result['arousal'],
                    'dominance': result['dominance'],
                    'continuous': result['continuous'],
                })

            overall_results = utils.calculate_prediction_results(results)

            return Response({
                "results": results,
                "overall_results": overall_results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def testrespred(request):
    context = {}

    if request.method == 'POST':
        # Handle the image upload from the form
        image_files = request.FILES.getlist('imageInput')  # Get all uploaded images

        print(f"Number of images uploaded: {len(image_files)}")

        if image_files:
            try:
                # Use the already loaded ResNet50 instance
                model = ResNet50

                # Prepare images for prediction
                images = []
                for image_file in image_files:
                    try:
                        image = Image.open(image_file)
                        images.append(image)
                    except Exception as img_open_err:
                        print(f"Error opening image {image_file.name}: {img_open_err}")
                        context['error'] = f"Could not open image {image_file.name}."
                        return render(request, 'testresnet.html', context)

                # Get predictions for the uploaded images
                results = model.predict(images)  # Call the predict method directly
                print(f"Predictions: {results}")

                # Prepare results for rendering
                results_to_display = [{
                    'filename': image_files[idx].name,
                    'arousal': results['arousal'],
                    'dominance': results['dominance'],
                    'continuous': results['continuous'],
                } for idx in range(len(images))]

                overall_results = utils.calculate_prediction_results(results_to_display)

                # Add the results to context to display on the page
                context['results'] = results_to_display
                context['overall_results'] = overall_results  # Include overall results for rendering

            except Exception as e:
                # If there's an error, add it to the context
                print(f"An error occurred during prediction: {e}")
                context['error'] = str(e)

        else:
            context['error'] = "No images were uploaded."

    return render(request, 'testresnet.html', context)




def home(request):

    return render(request, 'index.html')