# **Sentiment Analysis App**  

This project is a sentiment analysis web application built using Streamlit and a Logistic Regression model. It predicts whether a given text has a positive, negative, or neutral sentiment and visualizes the results with a pie chart.  

## **Features**  
Predicts sentiment using a Logistic Regression model  
Preprocesses text by removing stopwords  
Stores sentiment history and displays a pie chart  
Streamlit-based UI for easy interaction  
Option to reset sentiment history  

## **Installation & Usage**  

```sh
# 1. Clone the Repository
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Run the App
streamlit run src/mainpage.py
```
## **Usage Instructions**
Enter some text into the input box  
The app will classify the text and output the sentiment  
A pie chart will show your current and past sentiments  
Click reset to refresh the graph(0 all of the values)  

