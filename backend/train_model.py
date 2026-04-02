import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# Load and clean data
# Already cleaned CSV — load directly
df = pd.read_csv('data/fake_job_postings_cleaned.csv')
df['character_count'] = df['text'].apply(len)

# Features and labels
X = df[['telecommuting', 'ratio', 'text', 'character_count']]
y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=53)

# Train text model
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.text.values)
clf_text = SGDClassifier(loss='log_loss').fit(count_train, y_train)

# Train numeric model
X_train_num = X_train[['telecommuting', 'ratio', 'character_count']]
clf_num = SGDClassifier(loss='log_loss').fit(X_train_num, y_train)

# Save models
with open('model.pkl', 'wb') as f:
    pickle.dump({'vectorizer': count_vectorizer, 'clf_text': clf_text, 'clf_num': clf_num}, f)

print("Model saved to model.pkl")
