import pandas as pd
import joblib

# Load test data
df = pd.read_csv("tests/test_data.csv")

# Load your trained model
lr_model = joblib.load(open('data/logistic_regression_model.sav', 'rb'))

# Initialize Predicted_Sentiment column
df["Predicted_Sentiment"] = 0

# Loop through each tweet and predict sentiment
for i, tweet in enumerate(df["Tweet"]):
    prediction = lr_model.predict_proba([tweet])[0]
    prob_posi = prediction[1]
    prob_negi = prediction[0]

    # Assign sentiment based on the predicted probabilities
    if abs(prob_posi - prob_negi) < 0.2:
        df.at[i, "Predicted_Sentiment"] = 2  # Neutral sentiment
        print(f"Tweet: {tweet}\nNeutral statement\n")
    elif prob_negi > 0.5:
        df.at[i, "Predicted_Sentiment"] = 0  # Negative sentiment
        print(f"Tweet: {tweet}\nNegative sentiment\n")
    elif prob_posi > 0.5:
        df.at[i, "Predicted_Sentiment"] = 4  # Positive sentiment
        print(f"Tweet: {tweet}\nPositive sentiment\n")

# Compare results
print(df[["Tweet", "Sentiment", "Predicted_Sentiment"]])

count = 0
for i in range(len(df)):
    if df["Sentiment"][i] != df["Predicted_Sentiment"][i]:
        print(df["Tweet"][i])
        print(df["Sentiment"][i])
        print(df["Predicted_Sentiment"][i])
    else:
        count += 1  

print(count/len(df))

