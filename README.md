# Sentiment-Analysis

Text Cleaning using NLTK (Stopwords Removal & Lemmatization)

User reviews contain noise such as punctuation, numbers, and common words.
All text is converted to lowercase for consistency.
Special characters and digits are removed using regular expressions.
Stopwords (e.g., is, the, and) are removed using NLTK.
Lemmatization converts words to their base form (e.g., running → run).
This step ensures clean and meaningful text for modeling.

Sentiment Labeling based on Ratings (Positive / Neutral / Negative)
A rule-based approach is used to generate sentiment labels.
Rating-to-sentiment mapping:
4–5 → Positive
3 → Neutral
1–2 → Negative
The generated sentiment column is used as the target variable.
This avoids manual labeling and enables supervised learning.

TF-IDF (Unigrams + Bigrams) for Feature Extraction
Machine learning models cannot work directly with raw text.
TF-IDF converts cleaned text into numerical vectors.
Unigrams capture individual words.
Bigrams capture important word pairs (e.g., not good, very slow).
TF-IDF assigns higher weight to informative words and reduces noise.
This improves sentiment pattern detection.

Linear SVM (LinearSVC) for Sentiment Classification
A Linear Support Vector Machine is used for classification.
Performs well on high-dimensional sparse text data.
Efficient and fast for large datasets.
Learns decision boundaries between Positive, Neutral, and Negative sentiments.
Produces reliable predictions for unseen reviews.

Model Evaluation
Cross-Validation ensures model stability and reduces overfitting.
Classification Report evaluates:
Precision
Recall
F1-score
Confusion Matrix visualizes correct and incorrect predictions.
Confirms that the model generalizes well.

Exploratory Data Analysis (EDA)
The project answers real-world analytical questions:
Distribution of review ratings
Helpful vs non-helpful reviews
Positive vs negative keyword analysis
Platform comparison (Web vs Mobile)
Verified vs non-verified users
Review length vs rating
Location-based sentiment trends
ChatGPT version impact on sentiment

Word Cloud Analysis
Separate word clouds for:
Positive reviews
Negative reviews
1-star reviews
Helps visualize dominant themes and frequent complaints.
Useful for understanding user pain points.

Model & Artifact Persistence
Saved using pickle:
sentiment_model.pkl – trained ML model
tfidf.pkl – TF-IDF vectorizer
dashboard_data.pkl – processed dataset
stopwords.pkl – stopwords list
Allows reuse without retraining.

Streamlit Dashboard
Sentiment Predictor

User enters a review.
Text is cleaned and vectorized.
Model predicts sentiment in real time.

Analysis Dashboard
Overall sentiment distribution
Rating vs sentiment mismatch detection
Platform, location, version-based insights
Keyword frequency and negative themes
