# 📧 Spam Email Detection Using Machine Learning

## 📌 Project Overview

This project is a simple **Machine Learning-based Spam Email Detection system** that classifies email messages as either **Spam** or **Ham (Not Spam)**.

The project uses **Natural Language Processing (NLP)** techniques to convert email text into numerical features and then applies the **Multinomial Naive Bayes** algorithm for classification.

## 🎯 Objective

The main objective of this project is to build a machine learning model that can:

* Analyze email/message text
* Convert text into numerical features
* Learn patterns from the dataset
* Classify messages as **Spam** or **Ham**
* Evaluate the model using accuracy

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Natural Language Processing (NLP)
* Jupyter Notebook

## 🤖 Machine Learning Algorithm

### Multinomial Naive Bayes

The project uses the **Multinomial Naive Bayes** algorithm, which is commonly used for text classification problems such as spam detection.

## 🔄 Project Workflow

```text
Email Dataset
     ↓
Load Dataset using Pandas
     ↓
Extract Email Text
     ↓
Convert Text into Numerical Features
     ↓
CountVectorizer
     ↓
Train-Test Split (80/20)
     ↓
Multinomial Naive Bayes
     ↓
Model Prediction
     ↓
Accuracy Evaluation
     ↓
Spam / Ham Classification
```

## 📂 Dataset

The project uses an `emails.csv` dataset containing email/message text and a spam label.

The important columns used in the project are:

* `text` – Contains the email/message content
* `spam` – Indicates whether the message is spam (`1`) or harm (`0`)

## ⚙️ Implementation Steps

### 1. Import Required Libraries

The project imports Pandas and required Scikit-learn modules for vectorization, data splitting, model training, and evaluation.

### 2. Load the Dataset

The email dataset is loaded using Pandas.

```python
dataset = pd.read_csv('/emails.csv')
```

### 3. Convert Text into Numerical Features

`CountVectorizer` is used to convert the email text into numerical features that can be processed by the machine learning model.

```python
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(dataset['text'])
```

### 4. Split the Dataset

The dataset is divided into training and testing data using an **80/20 split**.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, dataset['spam'], test_size=0.2
)
```

### 5. Train the Model

A Multinomial Naive Bayes model is created and trained using the training data.

```python
model = MultinomialNB()
model.fit(x_train, y_train)
```

### 6. Make Predictions

The trained model predicts whether the test messages are spam or ham.

```python
yPred = model.predict(x_test)
```

### 7. Evaluate the Model

The model's accuracy is calculated using `accuracy_score`.

```python
accuracy = accuracy_score(y_test, yPred)
print(accuracy)
```

## 🔍 Message Prediction

The project also contains a function to classify a new message.

```python
def predictMessage(message):
    messageVector = vectorizer.transform([message])
    prediction = model.predict(messageVector)
    return 'Spam' if prediction[0] == 1 else 'Ham'
```

A new message can be passed to the function, and the model returns either:

* **Spam**
* **Harm**

## 📊 Model Evaluation

The project evaluates the trained model using **accuracy**.

The accuracy value is calculated from the predictions made on the test dataset.

> The exact accuracy depends on the dataset and train-test split used when running the notebook.

## 📁 Project Structure

```text
Spam-Email-Detection/
│
├── Spam_emali_detection_project.ipynb
├── emails.csv
└── README.md
```

## 🚀 How to Run the Project

1. Clone or download this repository.
2. Make sure Python is installed.
3. Install the required libraries:

```bash
pip install pandas scikit-learn
```

4. Place `emails.csv` in the expected project location.
5. Open `Spam_emali_detection_project.ipynb` in Jupyter Notebook or VS Code.
6. Run the cells in order.
7. Check the model accuracy and use the prediction function for new messages.

## 📚 What I Learned

Through this project, I learned about:

* Text classification
* Basic NLP
* CountVectorizer
* Train-test splitting
* Multinomial Naive Bayes
* Model prediction
* Accuracy evaluation
* Building a simple machine learning project using Python

## 🔮 Future Improvements

The project can be further improved by:

* Using a larger and more diverse dataset
* Applying text preprocessing techniques
* Trying TF-IDF vectorization
* Comparing multiple machine learning algorithms
* Evaluating precision, recall, and F1-score
* Creating a simple web interface for real-time prediction

## 👩‍💻 Author

**BTech Artificial Intelligence & Machine Learning Student**

This project was developed as part of my learning journey in **Machine Learning and Artificial Intelligence**.

