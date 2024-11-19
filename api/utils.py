from collections import Counter
import numpy as np # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore


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

# Keyword Extraction Function
def get_keywords(passage, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([passage])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:num_keywords]]
    return top_keywords


def calculate_prediction_results(results):
    """
    Calculate the final prediction results from model outputs.

    Args:
        results (list): A list of dictionaries containing arousal, dominance, and continuous results.
            Each dictionary has:
            - 'arousal': categorical prediction
            - 'dominance': categorical prediction
            - 'frustration': continuous prediction
            - 'mental_demand': continuous prediction
            - 'performance': continuous prediction
            - 'physical_demand': continuous prediction
            - 'effort': continuous prediction

    Returns:
        dict: A dictionary containing the most common arousal, most common dominance,
              and average continuous results.
    """
    # Extract arousal and dominance values as strings
    arousal_values = [result['arousal'] for result in results]
    dominance_values = [result['dominance'] for result in results]
    
    # Extract continuous results
    continuous_results = {
        'frustration': [],
        'mental_demand': [],
        'performance': [],
        'physical_demand': [],
        'effort': []
    }
    for result in results:
        continuous_results['frustration'].append(result['frustration'])
        continuous_results['mental_demand'].append(result['mental_demand'])
        continuous_results['performance'].append(result['performance'])
        continuous_results['physical_demand'].append(result['physical_demand'])
        continuous_results['effort'].append(result['effort'])

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
