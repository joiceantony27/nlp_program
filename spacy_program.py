import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Sample news article paragraph
text = "India launched its new weather satellite today to improve monsoon forecasting and disaster management. The launch was successful and marks a major milestone in the country’s space program."

# Define specific word(s) to remove
words_to_remove = ["india", "satellite"]

# Process the text
doc = nlp(text)

# Tokenization
tokens = [token.text for token in doc]
print("Tokens:", tokens)

# Lemmatization with stopword and specific word removal
lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.lower() not in words_to_remove]
print("After Lemmatization:", lemmas)
