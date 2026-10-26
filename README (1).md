# 🎭 Emotion Detection from Text

**ISE-2 Activity: Machine Learning (1ICPC406), ML Model Deployment**
Annasaheb Dange College of Engineering and Technology (ADCET), Ashta

**Live demo:** https://emotion-detection-hy7qd2qdatawmmxpxg5xnv.streamlit.app/

---

## 1. Objective

To build a machine learning model that classifies the emotion expressed in a short piece of English text into one of six classes (**joy, sadness, anger, fear, love, surprise**) and to deploy it as an interactive web application so that anyone can type a sentence and instantly see the predicted emotion along with class probabilities.

## 2. Team Members

| Name | Roll No. |
|------|----------|
| Pragati Kiran Patil | 4043 |
| Sanika Mahangade | <4041|
| Pranjal Mane | 4042 |
| Rushikesh Yesugade | 4044|

## 3. Software, Tools and Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| ML library | scikit-learn (TF-IDF vectorizer + Logistic Regression) |
| Data handling | pandas |
| Model storage | joblib (`emotion_model.joblib`) |
| Web app / deployment | Streamlit (Streamlit Community Cloud) |
| Dataset | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (6 emotion classes) |
| Version control | Git and GitHub |

## 4. Project Structure

```
emotion-detection/
├── streamlit_app.py        # Streamlit web app (main file for deployment)
├── emotion_model.joblib    # Trained model (TF-IDF + Logistic Regression) and labels
├── confusion_matrix.png    # Model evaluation result
├── requirements.txt        # Python dependencies
├── .devcontainer/          # Dev container configuration
└── README.md               # Project documentation
```

## 5. How It Works

1. Text is converted into numerical features using **TF-IDF**.
2. A **Logistic Regression** classifier predicts the probability of each of the 6 emotions.
3. The emotion with the highest probability is shown as the result, and a bar chart displays the probability of every class.

## 6. Step-by-Step Instructions for Execution

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/pragatipatil441/emotion-detection.git
cd emotion-detection

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run streamlit_app.py
```

The app opens automatically at **http://localhost:8501**.

### Using the app

1. Type or paste a sentence into the text box.
2. Click **Detect emotion**.
3. The predicted emotion, its confidence, and a probability bar chart are displayed.

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository, branch `main`, and set the main file to `streamlit_app.py`.
4. Click **Deploy** and copy the generated app link into the *Live demo* line at the top of this README.

## 7. Sample Input and Output

| Input sentence | Predicted emotion |
|----------------|-------------------|
| I am so happy today! | 😄 JOY |
| I feel very lonely and lost. | 😢 SADNESS |
| I can't believe he did this to me! | 😡 ANGER |
| I am terrified of the exam tomorrow. | 😨 FEAR |
| I adore spending time with my family. | ❤️ LOVE |
| Wow, I did not expect that at all! | 😲 SURPRISE |

> Replace the table above with the real outputs you observe from your running app.



## 8. Results

- **Test accuracy:** 8.98
- **Model:** TF-IDF + Logistic Regression
- **Classes:** joy, sadness, anger, fear, love, surprise
- The confusion matrix above shows which emotions are predicted correctly and which are commonly confused with each other (for example, *joy* and *love*).

## 9. Conclusion

The project shows that a simple and lightweight model (TF-IDF with Logistic Regression) can detect emotions in text with good accuracy, and that it can be deployed quickly as a user-friendly web app using Streamlit. The model is fast, needs no GPU, and gives probability scores for every emotion.

**Future improvements:** use transformer models such as DistilBERT for higher accuracy, support more emotion classes, and handle longer or multilingual text.

## 10. References

- dair-ai/emotion dataset: https://huggingface.co/datasets/dair-ai/emotion
- scikit-learn documentation: https://scikit-learn.org
- Streamlit documentation: https://docs.streamlit.io
