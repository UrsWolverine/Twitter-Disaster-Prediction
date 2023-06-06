# Twitter-Disaster-Prediction
We developed a project in the "CIS.600.M007.SPRING23.Prncpls: Social Media and Data Mining" course at Syracuse University that involved creating a Bidirectional LSTM model to classify whether a given tweet is about a real disaster or not.

# The requirements to run this project are:

## Dataset

1. The dataset is available on the Kaggle challenge Data Page accessible at this link here:[https://www.kaggle.com/c/nlp-getting-started/data/](https://www.kaggle.com/c/nlp-getting-started/data/).
2. The GloVetwitter27b100d file required for the second part of the project can be found here: [https://www.kaggle.com/datasets/bertcarremans/glovetwitter27b100dtxt](https://www.kaggle.com/datasets/bertcarremans/glovetwitter27b100dtxt)

## Installing pre-requisites

1. If using Anaconda navigator, it's recommended to create a conda environment steps for which can be found here: [https://conda.io/projects/conda/en/latest/user-guide/getting-started.html](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
2. To run this code, you will need to install the following libraries and packages: Numpy, pandas, matplotlib, seaborn, nltk, wordcloud, scikit-learn, genism, tqdm, keras, wordcloud
3. Additionally, you will need to download the following NLTK datasets: stopwords, wordnet, averaged\_perceptron\_tagger
4. You can install these packages using pip or conda package managers. Further install any libraries or packages whenever necessary.
5. Further as we did the project using conda environment, we can initiate the environment as stated in the example screenshot below (conda activate tf):

![](RackMultipart20230506-1-3szm5o_html_f8740c3eac8cc6f6.png)

1. Later open your Jupyter Notebook using the following command in the conda terminal – jupyter notebook

# Modules of the code –

## ML\_Traditional\_Algorithms\_Final.ipynb – 
The code in this file uses traditional machine learning algorithms to predict whether a tweet is about a real disaster or not. Two techniques were used for creating the Bag of Words model: CountVectorization and TF-IDF Vectorization. CountVectorization generates a vector of the length of the vocabulary with the count of each word in the tweet, while TF-IDF Vectorization generates a vector with the TF-IDF score of each word in the tweet. TF-IDF score indicates the relative importance of the word in all the tweets. High TF-IDF scores indicate that the word appears frequently in the tweets but not extremely frequently in all the tweets (e.g., stop words). The features created were used as input for traditional ML algorithms, such as SVM, Logistic Regression, and Multinomial Naive Bayes.

## GloVe\_LSTM\_Weighted\_Final.ipynb –
This file contains code for Twitter-Disaster-Prediction using GloVe Embedding, LSTM model & weighted probability score.

1. For the GloVe Embedding + LSTM model, we conducted basic text preprocessing steps without eliminating stop words since it might negatively affect the meaning of the tweets. We employed a pre-trained GloVe embedding layer that was trained on Twitter data, where each word was transformed into a 100-dimensional vector. The tweets were constrained to a maximum length of 32 words, and tweets exceeding this limit were truncated, while tweets with fewer words were filled with zero vectors. The generated vectors were then fed into a Bidirectional LSTM layer containing 32 LSTM cells. The results obtained from these cells were then passed through four Dense layers, and the final layer produced the probability of a tweet referring to a real disaster.

1. The GloVe Embedding + LSTM model + weighted probability score approach is similar to the previous approach mentioned, except for the probability step. We considered the percentage of real disaster tweets for each keyword associated with the tweet in the training data. We calculated the probability of a tweet being a real disaster tweet for the test data using the LSTM network with pre-trained GloVe embeddings. For each tweet that had an associated keyword, we took the average of the probability calculated by the LSTM network and the probability obtained by dividing the percentage of real disaster tweets for the keyword.

# Running the application –

1. Place all the dataset files obtained from the link stated in the Dataset section above, along with Traditional\_ML\_Algorithms\_Final.ipynb & LSTM\_GloVe\_Weighted\_Prob\_Final.ipynb in a single folder.
2. Open Jupyter Notebook, Google Colab or any other '.ipynb' file running application and open both the files - Traditional\_ML\_Algorithms\_Final.ipynb & LSTM\_GloVe\_Weighted\_Prob\_Final.ipynb respectively.
3. Run both the Notebooks, the model accuracy and related statistics can be seen. The final output obtained is saved in .csv file format with the following names - LSTM\_Glove\_aug\_weighted.csv, LSTM\_Glove\_non\_augmented.csv & submission.csv

Additional information regarding the techniques utilized and the outcomes attained can be accessed in the Presentation and Project Report files located in the files submitted.
