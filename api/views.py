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


# class ResNet50V2APIView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request):
#         # Handle the image upload from the form
#         image_files = request.FILES.getlist('images')  # Expecting multiple images

#         if not image_files:
#             return Response({"error": "No images were uploaded."}, status=status.HTTP_400_BAD_REQUEST)

#         results = []

#         try:
#             for image_file in image_files:
#                 image = Image.open(image_file)
#                 result = ResNet50.predict([image])  # Use the global instance
#                 results.append({
#                     'filename': image_file.name,
#                     'arousal': result['arousal'],
#                     'dominance': result['dominance'],
#                     'continuous': result['continuous'],
#                 })

#             overall_results = utils.calculate_prediction_results(results)

#             return Response({
#                 "results": results,
#                 "overall_results": overall_results
#             }, status=status.HTTP_200_OK)

#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# class ResNet50V2APIView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request):
#         """
#         Handle image uploads and return predictions for all classes.
#         """
#         # Handle the image upload from the form
#         image_files = request.FILES.getlist('images')  # Expecting multiple images

#         if not image_files:
#             return Response({"error": "No images were uploaded."}, status=status.HTTP_400_BAD_REQUEST)

#         results = []

#         try:
#             # Initialize the model
#             model = ResNet50

#             images = []
#             for image_file in image_files:
#                 try:
#                     image = Image.open(image_file)
#                     images.append(image)
#                 except Exception as e:
#                     return Response(
#                         {"error": f"Error processing image {image_file.name}: {str(e)}"},
#                         status=status.HTTP_400_BAD_REQUEST,
#                     )

#             # Predict using the ResNet50V2 model
#             predictions = model.predict(images)  # Using the predict method of the model

#             # Collect the results with filenames and predictions
#             for idx, prediction in enumerate(predictions):
#                 results.append({
#                     'filename': image_files[idx].name,
#                     'arousal': prediction['arousal'],
#                     'dominance': prediction['dominance'],
#                     'frustration': prediction['frustration'],
#                     'mental_demand': prediction['mental_demand'],
#                     'performance': prediction['performance'],
#                     'physical_demand': prediction['physical_demand'],
#                     'effort': prediction['effort'],
#                 })
        
#             return Response({
#                 "results": results,
#             }, status=status.HTTP_200_OK)

#         except Exception as e:
#             return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ResNet50V2APIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        """
        Handle image uploads and return predictions for all classes.
        """
        # Handle the image upload from the form
        image_files = request.FILES.getlist('images')  # Expecting multiple images

        if not image_files:
            return Response({"error": "No images were uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        try:
            # Initialize the model
            model = ResNet50

            images = []
            for image_file in image_files:
                try:
                    image = Image.open(image_file)
                    images.append(image)
                except Exception as e:
                    return Response(
                        {"error": f"Error processing image {image_file.name}: {str(e)}"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

            # Predict using the ResNet50V2 model
            predictions = model.predict(images)  # Using the predict method of the model

            # Collect the raw results with filenames and predictions
            for idx, prediction in enumerate(predictions):
                results.append({
                    'filename': image_files[idx].name,
                    'arousal': prediction['arousal'],
                    'dominance': prediction['dominance'],
                    'frustration': prediction['frustration'],
                    'mental_demand': prediction['mental_demand'],
                    'performance': prediction['performance'],
                    'physical_demand': prediction['physical_demand'],
                    'effort': prediction['effort'],
                })
        
            # Now, calculate the aggregated results
            aggregated_results = utils.calculate_prediction_results(predictions)

            # Return both the raw and aggregated results
            return Response({
                "results": results,
                "aggregated_results": aggregated_results  # Include aggregated results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# def testrespred(request):
#     context = {}

#     if request.method == 'POST':
#         # Handle the image upload from the form
#         image_files = request.FILES.getlist('imageInput')  # Get all uploaded images

#         if image_files:
#             try:
#                 # Prepare images for prediction
#                 images = []
#                 for image_file in image_files:
#                     try:
#                         image = Image.open(image_file)
#                         images.append(image)
#                     except Exception as img_open_err:
#                         context['error'] = f"Could not open image {image_file.name}."
#                         return render(request, 'testresnet.html', context)

#                 # Get predictions
#                 results = ResNet50.predict(images)


#                 # Prepare results for rendering
#                 results_to_display = [{
#                     'filename': image_files[idx].name,
#                     **result  # Unpack all predictions directly into the result dictionary
#                 } for idx, result in enumerate(results)]

#                 # Add the results to context
#                 context['results'] = results_to_display

#             except Exception as e:
#                 context['error'] = str(e)

#         else:
#             context['error'] = "No images were uploaded."

#     return render(request, 'testresnet.html', context)

def testrespred(request):
    context = {}

    if request.method == 'POST':
        # Handle the image upload from the form
        image_files = request.FILES.getlist('imageInput')  # Get all uploaded images

        if image_files:
            try:
                # Prepare images for prediction
                images = []
                for image_file in image_files:
                    try:
                        image = Image.open(image_file)
                        images.append(image)
                    except Exception as img_open_err:
                        context['error'] = f"Could not open image {image_file.name}."
                        return render(request, 'testresnet.html', context)

                # Get predictions
                model = ResNet50  # Initialize model
                results = model.predict(images)  # Using the predict method of the model

                # Prepare results for rendering
                results_to_display = [{
                    'filename': image_files[idx].name,
                    **result  # Unpack all predictions directly into the result dictionary
                } for idx, result in enumerate(results)]

                # Calculate aggregated results
                aggregated_results = utils.calculate_prediction_results(results)

                # Add both raw and aggregated results to context
                context['results'] = results_to_display
                context['aggregated_results'] = aggregated_results  # Add aggregated results

            except Exception as e:
                context['error'] = str(e)

        else:
            context['error'] = "No images were uploaded."

    return render(request, 'testresnet.html', context)


def home(request):

    return render(request, 'index.html')