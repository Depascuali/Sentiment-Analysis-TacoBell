# Sentiment-Analysis-TacoBell
Sentiment Analysis for Taco Bell reviews store located in Miami, FL.

# Introduction
In this project, we were encouraged to perform sentiment analysis on the reviews of any establishment on Google Maps. I chose to analyze a Taco Bell restaurant in Miami, FL. An establishment with an average review score (3.5 in this case) was selected to ensure a mix of negative and positive reviews. The objective is to analyze the comments, observe their evolution, and ultimately provide actionable tips to the business.

# Libraries
We import neccesary libraries for the case:
```python
from google.colab import drive
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
drive.mount('/content/drive')
```

# We connect to the data
```python
data_df = pd.read_excel('/content/drive/MyDrive/Courses/Specialization/Analytical Modelling/Dataset.xlsx', sheet_name='Sheet1')
```

# Initialize model and creation of sentiment analysis function
We initialize the sentiment analysis model.In this case, we are using SentimentIntensityAnalyzer from nltk library. We define the formula classify_sentiment to be used, as well as the thresholds for each classification:
```python
sia = SentimentIntensityAnalyzer()

def classify_sentiment(review):
    scores = sia.polarity_scores(review)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
```

# Apply new function and create column with classifications
We apply the function for each record, and create a column to allocate the classification:
```python
data_df['Review'] = data_df['Review'].astype(str)
data_df['Sentiment_Analysis'] = data_df['Review'].apply(classify_sentiment)
```

# Clean text in order to analyze most present words
As we desire to analyse the most present words in each type of review, we first create "limpiar_texto" function to clean review text. The process followed is:
- We first set the text to lower case.
- We join each word ignoring punctuation.
- We tokenize each word, ignoring stopwords.
(What is stopwords? Stop words are common words in a language (like "and", "the", "is") that are usually filtered out during text processing since they often don't contribute much meaning to the sentences for many analysis tasks.)
```python
stop_words = set(stopwords.words('english'))
```
```python
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join([char for char in texto if char not in string.punctuation])
    tokens = word_tokenize(texto)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens
```

# Separate the data set into 2 (One for Positive reviews, other for Negative reviews)
We separate the data in order to analyze most present words in each type of review
```python
positive_reviews_df = data_df[data_df['Sentiment_Analysis'] == 'Positive']
negative_reviews_df = data_df[data_df['Sentiment_Analysis'] == 'Negative']
```

# Join all reviews and recover TOP30 most present words (for each type of review)
We join all positive reviews and on the other side all negative reviews. We apply "limpiar_texto" function to tokenize words. Then, we count positive tokens and negative tokens and retrieve top 30 in each case. We also print results.
```python
positive_reviews_text = ' '.join(positive_reviews_df['Review'])
negative_reviews_text = ' '.join(negative_reviews_df['Review'])

positive_tokens = limpiar_texto(positive_reviews_text)
negative_tokens = limpiar_texto(negative_reviews_text)

positive_word_counts = Counter(positive_tokens)
negative_word_counts = Counter(negative_tokens)

most_common_positive = positive_word_counts.most_common(30)
most_common_negative = negative_word_counts.most_common(30)

print("Most Common Positive Words:")
for word, count in most_common_positive:
    print(f"{word}: {count}")

print("\nMost Common Negative Words:")
for word, count in most_common_negative:
    print(f"{word}: {count}")
```

# Create data sets to be analyzed on Tableau
We export the three data sets (General Data set, Positive Tokens count data set and Negative Tokens count data set. We then analyze results in Tableau.
```python
positive_df = pd.DataFrame(most_common_positive, columns=['Word', 'Frequency'])
negative_df = pd.DataFrame(most_common_negative, columns=['Word', 'Frequency'])

file_path = '/content/drive/My Drive/positive_reviews_words.xlsx'

# Exportar el DataFrame a un archivo Excel
positive_df.to_excel(file_path, index=False)

file_path = '/content/drive/My Drive/negative_reviews_words.xlsx'

# Exportar el DataFrame a un archivo Excel
negative_df.to_excel(file_path, index=False)

file_path = '/content/drive/My Drive/dataset.xlsx'

# Exportar el DataFrame a un archivo Excel
data_df.to_excel(file_path, index=False)
```


# Results
PD: The presentation is uploaded on this repository, feel free to download it and check out the results, here is a quick summary:
![image](https://github.com/Depascuali/Sentiment-Analysis-TacoBell/assets/97790973/642dbf5b-9e15-46aa-93e5-58bba3fa564f)

![image](https://github.com/Depascuali/Sentiment-Analysis-TacoBell/assets/97790973/da625bac-e807-4554-a04d-31d30fe22f75)

![image](https://github.com/Depascuali/Sentiment-Analysis-TacoBell/assets/97790973/ed6269e3-6fb7-41ad-ac12-e16b460d6845)


