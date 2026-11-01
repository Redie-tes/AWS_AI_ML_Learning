import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
df = pd.read_csv(
    r"E:\AI_ML\AI_MLWEEK3/data/SMSSpamCollection",
    sep ="\t",
    header= None,
    names=["labels", "emails"]
)
df.head()
emails = df["emails"]
labels = df["labels"]
print(df.shape)
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(emails, labels, test_size= 0.2, random_state = 42)
vec = TfidfVectorizer()
model = MultinomialNB()
x_train_vec = vec.fit_transform(x_train)
model.fit(x_train_vec, y_train)
x_test_vec = vec.transform(x_test)
y_pred = model.predict(x_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("model accuracy: ", accuracy)
trick_messages = [
    "Cl@im your fr33 g1ft now!",
    "Not free, just lunch with friends.",
    "Congratulations! You won a prize. Click here!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT!!! Claim your reward before midnight!!!"
]
trick_vectors = vec.transform(trick_messages)

predictions = model.predict(trick_vectors)

for message, prediction in zip(trick_messages, predictions):
    print("Message:", message)
    print("Prediction:", prediction)
    print("----------------------")