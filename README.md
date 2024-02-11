## Email Spam Detection

Email has become a ubiquitous tool for communication, saving time and cost. However, the rise of social networks has led to an increase in unwanted emails, commonly known as spam. Identifying and filtering out spam emails is crucial for maintaining the integrity and security of email communication. In this project, we leverage text classification techniques in Python to detect and classify email spam messages. We aim to assess the accuracy, processing time, and error rate of various algorithms (such as Naive Bayes, Multinomial Naive Bayes, and J48) on an Email Dataset, comparing their performance for text classification.

### Libraries Used
- `pickle`
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `string`
- `sklearn`
- `nltk`

### Project Pipeline

#### 1. Scoping
We collect and prepare the dataset for training. The goal is to identify spam emails using machine learning techniques, specifically algorithms like Naive Bayes and Multinomial.

#### 2. The Data
In this phase, we analyze and preprocess the dataset:

- Load the dataset
- Visualize dataset features frequencies
- Clean the data by handling missing values, checking for duplicates, removing punctuation, stopwords, and tokenizing it into words.

#### 3. Data Splitting
The dataset is split into training and testing sets, with 70% for training and 30% for testing.

#### 4. The Model
We create and train a Multinomial Naive Bayes model:

- Model creation and training
- Model evaluation using accuracy, classification report, and confusion matrix.
- Saving the trained model

#### 5. Using Gaussian Naive Bayes
We also explore the Gaussian Naive Bayes algorithm:

- Model creation and training
- Model evaluation using accuracy, classification report, and confusion matrix.
- Saving the trained model

#### 6. Comparing Scores
We compare the accuracy scores of both Multinomial and Gaussian Naive Bayes algorithms to assess their performance.

### References
- [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [J48 Algorithm](https://en.wikipedia.org/wiki/C4.5_algorithm)
