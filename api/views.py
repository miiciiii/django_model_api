# Importing Libraries

import os
from PIL import Image # type: ignore 

from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser


from .model_loader import initialize_models
from . import utils as utils

models = initialize_models()

# Access models
QG = models['QG']
AG = models['AG']
DG = models['DG']
ResNet50 = models['ResNet50']

if QG is None or AG is None or DG is None or ResNet50 is None:
    raise RuntimeError("RUNTIME ERROR : MODELS ARE NOT INITIALIZED")


#######################################################################################

#################### NATURAL LANGUAGE MODELS (T5) #########################

#######################################################################################



class PredictionAPIView(APIView):
    def post(self, request):
        passage_input = request.data.get('passage', None)

        if passage_input is None:
            return Response({"error": "No passage provided."}, status=status.HTTP_400_BAD_REQUEST)

        keywords = utils.get_keywords(passage_input, num_keywords=10)
        print(f"Extracted keywords: {keywords}")

        questions_and_answers_dict = {}


        if keywords:
            for idx, keyword in enumerate(keywords):
                generated_distractors_set = set()
                print(f"Processing keyword {idx + 1}: {keyword}")

                while len(questions_and_answers_dict) != 5:
                    generated_question = QG.generate(keyword, passage_input)
                    generated_answer = AG.generate(generated_question, passage_input)
                    generated_distractors = DG.generate(generated_question, generated_answer, passage_input)

                    print(f"Generated question: {generated_question}")
                    print(f"Generated answer: {generated_answer}")
                    print(f"Generated distractors: {generated_distractors}")

                    if len(generated_answer) >= 100:
                        generated_answer = keyword
                        print(f"Using keyword as answer: {generated_answer}")

                    for i in generated_distractors:
                        if i not in generated_distractors_set:
                            generated_distractors_set.add(i)
                    

                    max_attempts = 2
                    attempts = 0

                    while len(generated_distractors_set) < 3 and attempts < max_attempts:
                        for distractor in generated_distractors:
                            attempts += 1
                            new_distractor = DG.generate(generated_question, distractor, passage_input)
                            new_distractor_tuple = tuple(new_distractor)

                            for i in new_distractor_tuple:
                                if i not in generated_distractors_set:
                                    generated_distractors_set.add(i)

                            print(f"New distractor generated: {new_distractor}")

                            if len(generated_distractors_set) >= 3:
                                print("Sufficient distractors generated, breaking out of attempts loop.")
                                break

                            if attempts >= max_attempts:
                                print("Max attempts reached without enough distractors.")
                                break

                    if len(generated_distractors_set) >= 3:
                        generated_distractors = list(generated_distractors_set)
                        question_type = utils.classify_question_type(generated_question)
                        questions_and_answers_dict[f"question_{idx + 1}"] = {
                            "question": generated_question,
                            "choices": generated_distractors[:3],
                            "answer": generated_answer,
                            "question_type": question_type
                        }
                        print(f"Added question {idx + 1} to the dictionary.")
                        break
                    else:
                        print(f"Skipping question for '{keyword}' due to insufficient distractors.")
                        break

        else:
            questions_and_answers_dict["Invalid"] = {
                "question": "No question generated",
                "choices": [],
                "answer": "N/A",
                "question_type": "unknown"
            }

        response_data = {
            'passage': passage_input,
            'questions-choices-answer': questions_and_answers_dict
        }

        print("Final response data prepared.")
        return Response(response_data, status=status.HTTP_200_OK)



def testt5pred(request):
    random_passage_str = ''
    questions_and_answers_dict = {}

    if request.method == 'POST':
        random_passage_str = request.POST.get('passage', '')
        print(f"Received passage: . . .")  

        keywords = utils.get_keywords(random_passage_str, num_keywords=10)
        print(f"Extracted keywords: {keywords}") 

        if keywords:
            for idx, keyword in enumerate(keywords):
                generated_distractors_set = set()
                print(f"Processing keyword {idx + 1}: {keyword}")

                while len(questions_and_answers_dict) != 5:
                    generated_question = QG.generate(keyword, random_passage_str)
                    generated_answer = AG.generate(generated_question, random_passage_str)
                    generated_distractors = DG.generate(generated_question, generated_answer, random_passage_str)

                    print(f"Generated question: {generated_question}")
                    print(f"Generated answer: {generated_answer}")
                    print(f"Generated distractors: {generated_distractors}")

                    print(f"KASNDKJASJKDJKASDJBSD : {len(generated_answer)}")

                    if len(generated_answer) >= 100:
                        generated_answer = keyword
                        print(f"Using keyword as answer: {generated_answer}")

                    for i in generated_distractors:
                        if i not in generated_distractors_set:
                            generated_distractors_set.add(i)
                    

                    max_attempts = 2
                    attempts = 0

                    while len(generated_distractors_set) < 3 and attempts < max_attempts:
                        for distractor in generated_distractors:
                            attempts += 1
                            new_distractor = DG.generate(generated_question, distractor, random_passage_str)
                            new_distractor_tuple = tuple(new_distractor)

                            for i in new_distractor_tuple:
                                if i not in generated_distractors_set:
                                    generated_distractors_set.add(i)

                            print(f"New distractor generated: {new_distractor}")

                            if len(generated_distractors_set) >= 3:
                                print("Sufficient distractors generated, breaking out of attempts loop.")
                                break

                            if attempts >= max_attempts:
                                print("Max attempts reached without enough distractors.")
                                break

                    if len(generated_distractors_set) >= 3:
                        generated_distractors = list(generated_distractors_set)
                        question_type = utils.classify_question_type(generated_question)
                        questions_and_answers_dict[f"question_{idx + 1}"] = {
                            "question": generated_question,
                            "choices": generated_distractors[:3],
                            "answer": generated_answer,
                            "question_type": question_type
                        }
                        print(f"Added question {idx + 1} to the dictionary.")
                        break
                    else:
                        print(f"Skipping question for '{keyword}' due to insufficient distractors.")
                        break

        else:
            questions_and_answers_dict["Invalid"] = {
                "question": "No question generated",
                "choices": [],
                "answer": "N/A",
                "question_type": "unknown"
            }


    # Prepare context for rendering
    context = {
        'passage': random_passage_str,
        'questions_choices_answer': questions_and_answers_dict
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