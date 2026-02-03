Project Workflow
House price prediction
Data Cleaning Feature Selection:
Dropped columns that didn't significantly impact price (e.g.,area_type,availability,society,balcony). Handling Null Values: Removed rows with missing values as they represented a very small percentage of the total dataset.

Feature Engineering BHK Creation:
Extracted the number of bedrooms from the size column and converted it to an integer. Total Square Feet Normalization: Handled the total_sqft column by: Converting single values to floats. Taking the average of range values (e.g., '2100 - 2850'). Removing non-standard units (e.g., 'Sq. Meter'). Price per Sqft: Created a new feature price_per_sqft to help identify price anomalies.

Dimensionality Reduction Location Tagging:
There were over 1,200 unique locations. Any location with fewer than 10 data points was tagged as 'other'. This prevents the "Curse of Dimensionality" when performing One-Hot Encoding later.

Tech Stack 
Python Pandas (Data Manipulation) 
NumPy (Mathematical Operations)
Matplotlib (Data Visualization)

How to Run Upload the bengaluru_house_prices.csv to your Google Colab environment. Run the cells sequentially to see the data transformation.


#### spam detection

SMS Spam Detection
Project This project is a Machine Learning-based SMS Spam Classifier that identifies whether a text message is Ham (legitimate) or Spam. It features a full pipeline from raw data cleaning to model deployment readiness.

Project Overview 
The goal was to build a highly precise model. In spam detection, Precision is more important than Accuracy because we want to avoid "False Positives" (marking a real message as spam).

Data Cleaning Removed unnecessary columns and handled null values. Encoded labels: 0 for Ham and 1 for Spam. Removed duplicate entries to prevent model bias.
2 Exploratory Data Analysis (EDA) Analyzed the distribution of characters, words, and sentences using nltk. Insight: Spam messages tend to have a higher number of characters and words compared to ham messages. Visualized data using Pie charts, Histograms, and Heatmaps.

Text Preprocessing A custom transform_text function was built to: Convert text to Lowercase. Tokenize sentences into individual words. Remove Special Characters and Punctuation.
Filter out Stopwords (common words like "the", "is", etc.).

Apply Stemming (e.g., "loving" → "love") using the PorterStemmer.

Model Building & Evaluation We tested three variants of Naive Bayes using both CountVectorizer and TfidfVectorizer: GaussianNB MultinomialNB BernoulliNB The Winner: TfidfVectorizer + MultinomialNB Accuracy: ~97% Precision: 1.0 (100%) — This ensures no legitimate messages are incorrectly flagged as spam.
Tech Stack Language: Python Libraries: Pandas, NumPy, Scikit-learn, NLTK, Seaborn, Matplotlib

Tools: WordCloud (for keyword visualization), Pickle (for model export)

Files for Deployment vectorizer.pkl: The saved TF-IDF vectorizer. 
model.pkl: The trained Multinomial Naive Bayes model.
