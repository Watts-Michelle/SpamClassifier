# Spam classifier using Naive Bayes

## What is Spam?

Spam is any kind of unwanted emails messages, that are usually scams or unsolicited advertisements.

There are many techniques used to fight spam. Most major email service providers have spam detection systems built in that will automatically classify spam mail as “junk” mail.

We will take a look at a popular technique of e-mail filtering called Naïve Bayes and how it can be used to create an email classifier.

## What is Naive Bayes?

In short, the Bayes theorem calculates the probability of a certain event happening (in our case, a message being spam) given that a certain other event has happened (in our case, the appearance of certain words in a message). It assumes that predictors are independent, hence it is called naïve. 

## Naive Bayes Implementation

### Dataset

In our implementation each email message will be represented as a “bag of words”. Which is to say that a count will be taken of the frequency of each word in a message.

It will consist of binary variables, 0 to represent “ham” and 1 to represent “spam”. Ham being a term used to mean an email that is not “spam”.

### Calculate Prior Probabilities

First we calculate the prior probabilities, by counting the number of spam messages and the number of ham messages and dividing both by the total number of messages.

```python
    def estimate_log_class_priors(self, data):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column, calculate the logarithm of the empirical class priors,
        that is, the logarithm of the proportions of 0s and 1s:
            log(p(C=0)) and log(p(C=1))

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                     the first column contains the binary response (coded as 0s and 1s).

        :return log_class_priors: a numpy array of length two
        """
        total_data_size = len(data)
        ham = 0
        spam = 0

        for i in data:
            if i[0] == 0: ham += 1
            else: spam += 1

        ham_prior = np.log(ham / total_data_size)
        spam_prior = np.log(spam / total_data_size)

        return np.array([spam_prior, ham_prior])
```

### Calculate Keyword Probabilities

The next step is to calculate, for each keyword in the vocabulary, the probability that it occurs if this is a spam or ham message.

This will decide the messages classification. We use Laplace smoothing here to handle the problem of zero probability.

```python
    def estimate_log_class_conditional_likelihoods(self, data, alpha=1.0):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column and binary features (words), calculate the empirical
        class-conditional likelihoods, that is,
        log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

        Assume a multinomial feature distribution and use Laplace smoothing
        if alpha > 0.

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

        :return theta:
            a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging
            to class j.
        """
        spam = data[data[:, 0] == 1]
        ham = data[data[:, 0] != 1]

        spam_sums = []
        for i in range(1, spam.shape[1]):
            spam_sums.append(spam[:, i].sum() )

        ham_sums = []
        for i in range(1, ham.shape[1]):
            ham_sums.append(ham[:, i].sum())

        spam_total = sum(spam_sums)
        ham_total = sum(ham_sums)

        for i in range(0, len(spam_sums)):
            spam_sums[i] = (spam_sums[i] + alpha) / (spam_total + (len(spam_sums) + alpha))

        for i in range(len(ham_sums)):
            ham_sums[i] = (ham_sums[i] + alpha) / (ham_total + (len(ham_sums) + alpha))

        return np.array([np.log(spam_sums), np.log(ham_sums)])
```

### Training The Model

To train the model we load our dataset and calculate the prior probabilities and the probabilities of individual keywords.
```python
    def train(self):
        training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)

        self.log_class_priors = self.estimate_log_class_priors(training_spam)
        self.log_class_conditional_likelihoods = self.estimate_log_class_conditional_likelihoods(training_spam)
```

### Classifying Messages

To train the model we simply load the training dataset and calculate the prior probabilities and the probabilities of individual keywords based on the class:

```python
    def predict(self, new_data):
        """
        Given a new data set with binary features, predict the corresponding
        response for each instance (row) of the new_data set.

        :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
        :param log_class_priors: a numpy array of length 2.
        :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
            theta[j, i] corresponds to the logarithm of the probability of feature i appearing
            in a sample belonging to class j.
        :return class_predictions: a numpy array containing the class predictions for each row
            of new_data.
        """
        class_predictions = []

        for i in new_data:
            spam_probability = self.log_class_priors[0] + (np.sum(self.log_class_conditional_likelihoods[0] * i))
            ham_probability = self.log_class_priors[1] + (np.sum(self.log_class_conditional_likelihoods[1] * i))
            class_predictions.append(np.argmax(np.array([ham_probability, spam_probability])))

        return np.array(class_predictions)
```

## Accuracy

This model results in an accuracy of 89%.
