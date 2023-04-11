# Real and Fake News detection (Data Science Project)

(A) What is a fake news?

A type of yellow journalism. Fake news is false or misleading information presented as news. Fake news often has the aim of damaging the reputation of a person or entity, or making money through advertising revenue. This is often done to further or impose certain ideas and is often achieved with political agendas. Such news items may contain false and/or exaggerated claims, and may end up being viralized by algorithms, and users may end up in a filter bubble.

(B) What is a TfidfVectorizer?
TF:

(Term Frequency): The number of times a word appears in a document is its Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.
IDF:

(Inverse Document Frequency): Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus.



TF-IDF Vectorizer is a measure of originality of a word by comparing the number of times a word appears in document with the number of documents the word appears in. formula for TF-IDF is:

TF-IDF = TF(t, d) x IDF(t), where, TF(t, d) = Number of times term "t" appears in a document "d". IDF(t) = Inverse document frequency of the term t.

The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

C) What is a PassiveAggressiveClassifier?

The passive aggressive classifier is a machine learning algorithm that is used for classification tasks.
Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.



(D)  The Dataset

The dataset we’ll use for this python project is news.csv. This dataset has a shape of 7796×4. 

      The first column identifies the news, 
      the second,
      third are the title and text, 
      and the fourth column has labels denoting whether the news is REAL or FAKE. The dataset takes up 29.2MB of space and you can download it here.[news.zip](https://github.com/Harkeerat-Pathak/real_fake_news_detection_datascienceproject/files/11196495/news.zip)


## Python Code and screenshots  for fake and real news detection 


STEP 1: IMPORTING THE NECESSARY LIBRARIES

![libraries](https://user-images.githubusercontent.com/69766918/231121759-38667a3b-03c3-4cff-b5e1-bcd984a29012.jpg)

STEP 2: Reading the CSV file into dataframe

![calling csv](https://user-images.githubusercontent.com/69766918/231126000-d434d270-3c43-4ae1-8302-e5994e0a25c9.jpg)


STEP 3: Checking shape, head and tail
        
        (a) Shape
![shape of the database](https://user-images.githubusercontent.com/69766918/231127272-ba1eaa51-0ea2-4c7e-b87e-e1ab57fa0d90.jpg)

        (b) head(reading first 5 rows)
![head](https://user-images.githubusercontent.com/69766918/231129708-6f8a6086-5f88-49fc-8cd5-41de8db2a1a6.jpg)

     
        (c) Tail(reading last 5 rows)
        
![tail](https://user-images.githubusercontent.com/69766918/231128628-d7940475-5202-4497-89aa-4f10eab8311e.jpg)


STEP 4:(a) Checking number of real and fake counts

![value count of fake and real](https://user-images.githubusercontent.com/69766918/231132227-014fd626-fb94-4f9d-916b-2ca7f528f6be.jpg)

(b) #create bar plot to visualize frequency of each team
![barplot-fake-real](https://user-images.githubusercontent.com/69766918/231139864-37fe3952-9508-4c65-93d5-d23ef1958043.jpg)


STEP 5: Split the dataset into training and testing sets.
        
![train-test-split](https://user-images.githubusercontent.com/69766918/231135623-51674497-921e-429f-862f-134057c51d62.jpg)

STEP 6: Let’s initialize a TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7 (terms with a higher document frequency will be discarded). Stop words are the most common words in a language that are to be filtered out before processing the natural language data. And a TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.
![tfidvectorizer](https://user-images.githubusercontent.com/69766918/231141525-30ea2123-ffaa-438f-8047-1f054be83da8.jpg)

STEP 7: initialize a PassiveAggressiveClassifier

Next, we’ll initialize a PassiveAggressiveClassifier. We’ll fit this on tfidf_train and y_train.
Then, we’ll predict on the test set from the TfidfVectorizer and calculate the accuracy with accuracy_score() from sklearn.metrics.

![accuracy](https://user-images.githubusercontent.com/69766918/231143310-3d79514b-cf71-46fb-b5d5-70a90f806df3.jpg)

STEP 8: Finally, let’s print out a confusion matrix to gain insight into the number of false and true negatives and positives.
![confusion matrix](https://user-images.githubusercontent.com/69766918/231143564-619d1216-7a45-4c82-ab29-63b2b3359f6a.jpg)

## Summary

We eneded up with accuracy score nearly upto 92%. I throughly enjoyed the fake news detection python project.

Although I must say there were mistakes while running the code. As you can see in the screenshots also that steps for the news detection are not in sequential order. Sometimes I have to insert cells above or below the code and that has changed the sequence number too.

Happy Learning!
