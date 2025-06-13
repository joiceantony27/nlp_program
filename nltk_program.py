import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample news article paragraph
text = "India launched its new weather satellite today to improve monsoon forecasting and disaster management. The launch was successful and marks a major milestone in the country’s space program."

# Define specific word(s) to remove
words_to_remove = ["india", "satellite"]

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Stopword + specific word removal
filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english') and word.lower() not in words_to_remove]
print("After Stopword & Word Removal:", filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("After Stemming:", stemmed_tokens)
