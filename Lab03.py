""""""
#Exercise 3.1
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk import download

# Download necessary resources
download('punkt')
download('averaged_perceptron_tagger')
download('maxent_ne_chunker')
download('words')

# Sample text
text1 = """
When Sebastian Thrun started working on self-driving cars at 
Google in 2007, few people outside of the company took him 
seriously. “I can tell you very senior CEOs of major American 
car companies would shake my hand and turn away because I wasn’t 
worth talking to,” said Thrun, in an interview with Recode earlier 
this week.
"""
text2 = """
When olivia Martinez started working her research on quantum computing at Quantum Labs in 2018,
many industry leaders were dismissive. I recall senior executives from leading software firms 
politely nodding and walking away because they didn't see the potential, Martinez reflected in an interview 
with Innovation Weekly earlier this year.
"""

text3 = """
In 2010, when Jack Robinson was pioneering new AI algorithms at NeuraTech Solutions,
most people outside the company dismissed the idea as impractical. I met with high-ranking officials at 
major tech corporations who would offer a handshake and quickly change the subject, thinking 
it was just a passing trend, Robinson shared in an interview with AI Today last month.
"""
Text4 = """
In 2018, when Daniel Hayes was revolutionizing blockchain security at CipherTech Labs, 
many industry leaders dismissed the concept as overhyped. I engaged with influential 
figures in top financial institutions who would smile politely and quickly steer the 
conversation elsewhere, convinced it was just another passing fad, Hayes recounted in 
an interview with FinTech Insights last quarter.
"""



# Tokenize the text into sentences
sentences = sent_tokenize(Text4)

# Tokenize each sentence into words and apply POS tagging
words = word_tokenize(Text4)
pos_tags = pos_tag(words)

# Find noun phrases (noun chunks) - NLTK doesn't have a direct equivalent to Spacy's noun_chunks
# We can use POS tags to extract noun phrases
#noun_phrases = [chunk[0] for chunk in pos_tags if chunk[1] in ['NN', 'NNS', 'NNP', 'NNPS']]

# Named Entity Recognition (NER)
named_entities = ne_chunk(pos_tags)



"""print("Named Entities:")
for entity in named_entities:
    if isinstance(entity, tuple):
        continue  # Skip non-entity parts of speech
    print(" ".join(c[0] for c in entity), entity.label)"""

#Exercise 3.3
"""def identify_language(text):
    # Define common words in each language
    english_words = {"the", "a", "and", "of", "be", "that", "have", "it", "for", "not"}
    french_words = {"le", "la", "de", "ne", "et", "un", "pas", "vous", "etre", "les", "en"}
    spanish_words = {"el", "la", "de", "no", "y", "un", "por", "qué", "me", "una", "los"}
    twi_words = {"me", "wo", "ɔno", "yɛ", "mo", "sɛ", "deɛ", "ne", "na", "bi", "pa", "nye"}
    swahili_words = {"na", "ya", "kwa", "ni", "wewe", "mimi", "lakini", "yeye", "sisi", "hao", "huyu"}

 # Split the input text into words
    words = text.lower().split()

    words = text.lower().split()
    scores = {"English": 0, "French": 0, "Spanish": 0, "Swahili": 0, "Twi": 0}


    # Count matches for each language
    for word in words:
        if word in english_words:
            scores["English"] += 1
        if word in french_words:
            scores["French"] += 1
        if word in spanish_words:
            scores["Spanish"] += 1
        if word in twi_words:
            scores["Twi"] += 1
        if word in swahili_words:
            scores["Swahili"] += 1


    # Find the language with the highest score
    detected_language = max(scores, key=scores.get)
    
    # Return the detected language
    return detected_language if scores[detected_language] > 0 else "Unknown"


# Test sentences (test set)
test_sentences = [
    ("I am going to the store to buy some groceries.", "English"),
    ("It is a beautiful day outside.", "English"),
    ("J'aime beaucoup la musique classique", "French"),
    ("Comment allez-vous aujourd'hui", "French"),
    ("Me ho yɛ fɛ, na m'ani agye.", "Twi"),
    ("Habari yako, leo ni siku nzuri.", "Swahili"),
    ("Voy a la tienda a comprar pan.", "Spanish"),
    ("Vous avez vu ce film hier soir?", "French"),
    ("Hace mucho calor en el verano.", "Spanish")
]

# Initialize variables for accuracy calculation
correct_predictions = 0
total_sentences = len(test_sentences)

# Run each test sentence through the language identification system
for sentence, actual_language in test_sentences:
    predicted_language = identify_language(sentence)
    print(f"Sentence: {sentence}")
    print(f"Predicted Language: {predicted_language}")
    print(f"Actual Language: {actual_language}")
    print("-------------------------------")

    # Check if the prediction was correct
    if predicted_language == actual_language:
        correct_predictions += 1

# Calculate accuracy
accuracy = (correct_predictions / total_sentences) * 100
print(f"\nAccuracy: {accuracy}%")"""

#3.4
#import libraries
import pandas as pd
import nltk
nltk.download ('all')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the amazon review dataset (using sample in Blackboard)
df = pd.read_csv ('amazon.csv')
df
