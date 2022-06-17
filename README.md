# TWITTER-SENTIMENT-ANALYSIS-FOR-CRYPTOCURRENCIES ( DATA COLLECTION)
The Twitter Application Programming Interface (API) allows users, businesses, and developers to have automated access to the public data that Twitter users have given consent to disclose with the rest of the world. This information includes tweets, user profiles, and other publicly available information.

TWITTER DEVELOPER ACCOUNT:
It is necessary to have a Twitter developer account to collect tweets from other users. To apply for a developer account on Twitter, connect to developer.twitter.com, while logged into your existing Twitter account. Follow the on-screen instructions to complete the application. After clicking the sign-up button, enter the name of the developer account, the location, and the details of the use case. Check the developer agreement, make sure it's acceptable, and then submit it. After signing up for a Twitter developer account, which is completely free of charge and grants rapid access to the Twitter API, we are granted "essential access."

ESSENTIAL ACCESS:
There are now two additional access levels that each provide you with more data at no additional cost.
Essential access grants the user immediate access to the Twitter API version 2, one App environment, and the potential to retrieve up to 500k Tweets each month. This ought to satisfy the requirements of most developers, particularly in the beginning stages. 

ELEVATED ACCESS:
So, we need to apply for elevated access. Explain why you need an elevated access for your use case properly mentioning that you’re a student and the clear intention is to research for a thesis project. This access including the capability to retrieve up to two million Tweets every single month and three different App environments (development, staging, and production). If you are already using Twitter API version 2, you will see an upgrade to Elevated access for your Projects today. This will take place automatically. In that case, you will be required to apply for Elevated access. Twitter is working on a way to enable developers who have previously had their requests for a developer account denied applying for an account again. Soon, it will provide a mechanism that enables reapplication for Elevated access or signup for Essential access. Twitter API has different credentials compared to other API’s. API keys are used to build the apps and projects in developer portal. 

CONSUMER KEYS:
There are four types of secret keys and tokens which are unique to every user. Think of these as the username and password that represents the app when making API requests.

AUTHENTICATION TOKENS:
BEARER TOKEN: Bearer token authenticates requests on behalf of the developer app.

ACCESS TOKENS AND SECRET KEYS:
To authenticate OAuth 1.0a API, users must have an Access Token and a Secret. They name the Twitter account on whose behalf the request is made.

CHOICE OF CRYPTOCURRENCY: Bitcoin
Cryptocurrencies such as Bitcoin (BTC) attracted a lot of attention in recent months due to their unprecedented price fluctuations.

CHOICE OF INFLUENCERS: Influencers selection has been made based on the number of followers.
1.	Anthony Pompliano.                       
2.	Andreas M. Antonopoulos
3.	Roger Keith Ver
4.	Barry Silbert
5.	Vitalik Buterin
6.	Michael J. Saylor
7.	Bobby Lee
8.	Charlie Lee
9.	Jack Dorsey
10.	Bitcoin Archive
11.	Tyler Winklevoss
12.	Cameron Winklevoss
13.	Scott Melker
14.	Lex Moskovski

Three columns in this data set are tweets, date and time of the tweet posted and influencer name.
In the cleaning process the timestamp is removed and converted into the same format as the date column present BTC prices dataset.

BTC PRICES:
Using yfinance library BTC-GBP prices from 2014 to 2022 are downloaded.
The columns in this dataset are date, open, high, low, close, Adjclose, Volume.

DATA CLEANING AND PREPROCESSING:
LIBRARIES USED:
•	NumPy
•	Re
•	Matplotlib
•	Textblob
•	Wordcloud

In the cleaning process and analysis only three columns are considered, 
• i.e Date, Open, Close by using the “drop” function removed the unwanted columns.
•	Merging of two datasets is needed to make it work for analysis and it is done by “joins”.
•	Before merging the datasets, make sure the datatype of the date column must be same.
•	To merge the two different tables in python, there is a concept called “joins”.
•	There are four types of joins inner join, full join(outer), left join and right join.
•	The results of a Full Join, which is often referred to as a Full Outer Join, include all the entries that either have a match in the left or right data frame.
•	When rows in both data frames do not match, the resulting data frame will include NaN for every column of the data frame that does not have a matching row. This is because NaN stands for not a number.
•	To do a Full join, we need to do nothing more than tell the merge () function that it’s how argument should be set to "outer."
•	Check if there are any null values after joining the columns by using isnull () function and remove them by using dropna ().

FEATURE ENGINEERING: The process of selecting, altering, and transforming raw data into features that may be used in supervised learning is referred to as feature engineering. This process is also known as feature extraction.
•	Few columns are added by using the existing columns.
•	Price is calculated by subtracting closing price from opening price. 
•	Sent_price is another column added based on the price column which says whether it has increased or decreased.
•	Using regular expressions, the unwanted symbols like #, @, RT, white spaces, hyperlinks are removed.
•	New column is created with the name called “cleaned_tweets”.

POLARITY: The intensity of a person's viewpoint is referred to as their polarity. It is possible for it to be either positive or negative. The value of polarity is a float, which can fall anywhere between -1 and 1, with 1 denoting a positive statement and -1 indicating a negative statement. 

SUBJECTIVITY: The degree to which a person is personally connected with an object is referred to as their level of subjectivity toward that thing. Subjective sentences typically refer to an individual's opinion, feeling, or judgement, whereas objective sentences refer to material that is factual in nature. Subjectivity is also a float that can take any value between 0 and 1, as shown in the range [0,1].

Wordcloud: A word cloud is a visual representation of text data that is also sometimes referred to as a tag cloud or weighted list. Words are often single words, and the importance of each word is indicated by the size of the font or the colour of the font. The most often used expression has its characters enlarged and bolded, and it is presented in a variety of colours. The less significant a term is, the more it recedes into the background.

MODEL DEVELOPMENT:

This section will describe the methodology that should be used when developing models and selecting models to use. This also comprises the findings obtained from training and testing the model, selecting a baseline, and hyper-tuning the model that was chosen to achieve the highest possible level of accuracy.

LIBRARIES USED: Keras and TensorFlow

STEPS INVOLVED IN MODEL BUILDING:
•	Define network - In Keras, the definition of a neural network is represented as a stack of layers. The Sequential class serves as the container for these several layers.
Creating an instance of the Sequential class is the first thing that needs to be done. After that, you can start creating your layers and adding them in the correct sequence so that they can be connected. The recurrent layer known as LSTM that is made up of memory units is referred to as LSTM (). Dense refers to a fully linked layer that frequently comes after LSTM layers and is employed for the purpose of producing a prediction (). 

Adding additional LSTM layers to a Sequential model allows for the layers to be stacked. When stacking LSTM layers, it is critical that we output a sequence rather than a single value for each input. This is necessary so that the succeeding LSTM layer may obtain the necessary 3D input. This can be accomplished by directing the return sequences option to always return true values. Imagine that a Sequential model is a pipeline, with the raw data going in one end and the predictions coming out the other. This is a valuable container in Keras since it allows for worries that were traditionally connected with a layer to also be broken off and placed as distinct layers. This makes it easier to see the role that these concerns play in the transformation of data from input to prediction.

Activation functions, for instance, can be extracted and added to the Sequential as a layer-like object named Activation. These functions alter a summed signal from each neuron in a layer.

•	Compilation of the network - The compilation process is an efficient phase. It takes the straightforward order of layers that we defined and converts it into a format that can either be executed on your graphics processing unit (GPU) or your central processing unit (CPU), depending on how Keras is configured. This results in a highly efficient series of matrix transformations. Compilation can be considered of as a first step in the computation process for your network. It is usually necessary to do so after the definition of a model.

To compile something, you need to specify several parameters that are uniquely adapted to the process of training your network. In particular, the optimization method that should be used to train the network, as well as the loss function that should be used to evaluate the network, and which should be minimised by the optimization algorithm.

•	Fit the network - The network can then be "fit," which means the weights can be adjusted based on a training dataset, after it has been produced. The training data, which consist of a matrix of input patterns denoted by X and an array of output patterns that match those inputs, must be specified before the network can be fitted.

During the modelling process, the backpropagation method is used to train the network, and the optimization algorithm and loss function that were defined at the time of model compilation are used to guide the network's optimization. The backpropagation algorithm mandates that the network be trained for a predetermined number of iterations, often known as exposures to the training dataset.

Each epoch can be divided up into different groupings of input-output pattern pairs that are referred to as batches. This determines the number of different patterns that a network is presented with before the weights are changed during an epoch. Additionally, it is an optimization for efficiency, as it ensures that not an excessive number of input patterns are loaded into memory all at once.

•	Evaluation - After the training phase is complete, the network will be ready for analysis. It is possible to evaluate the network using the training data; however, this will not provide a helpful indicator of the effectiveness of the network as a predictive model because the network has already been exposed to all the training data. On a distinct dataset, one that was hidden from view during testing, we can judge how well the network performs. 

This will provide an approximation of the performance of the network when it comes to making predictions for data that has not yet been seen in the future. The model does an analysis on the loss experienced across all the test patterns, in addition to analysing the classification accuracy and any other metrics that were supplied while the model was being created. The response includes a list of metric indicators for review.


•	Predictions - When we have reached a point where we are satisfied with how well our model fits the data, we can start using it to make predictions based on newly collected information. This can be accomplished by simply invoking the models predict () function with an input pattern array and doing the analysis.

TOKENIZATION:

Keras offers a more advanced application programming interface (API) for creating text that can be adapted to numerous text documents and reused to do so.

For preparing text documents for deep learning, the Tokenizer class is made available by Keras. Constructing the Tokenizer is necessary, followed by testing its functionality on either raw text documents or integer encoded text documents.

The raw text is split into words and sentences that are referred to as tokens via tokenization. Importing the Tokenizer module that is in the keras.preprocessing.text package will accomplish this goal. By analysing the order in which the words appear in the text, tokenization helps in deciphering the meaning of the text.

PAD SEQUENCING:
Following the tokenization process, pad sequence needs to be completed. pad sequences are a function that is utilised to make sure that all the sequences contained inside a list are of the same length. This is accomplished by replacing 0 at the beginning of each sequence, as a default, until the length of each sequence is equal to the length of the longest sequence.

ONE-HOT ENCODING: One hot encoding may be characterised as the crucial process of changing the categorical data variables to be delivered to machine and deep learning algorithms, which in turn increase predictions as well as the classification accuracy of a model. The pre-processing of categorical features for use in machine learning models is frequently accomplished using the One Hot Encoding technique.

During the process, it takes a column that has category data that has been label encoded and then splits the column that comes after it into numerous columns.   Depending on which column contains a value, the numbers are mixed and replaced with random ones and zeros.
•	Get_dummies () are one of the one hot encoding function which performs the encoding on the columns we pass through it.
•	Before building the model, Sentiment column is passed to get_dummies so that the words positive, negative, and neutral is converted into 1’s and 0’s format.

•	EARLYSTOPPING: Inadequate training will result in the model having a underfit to both the train and test sets. If the model receives an excessive amount of training, it will overfit the training dataset and perform poorly when tested on a separate test dataset.

•	During the training of the network, a greater number of training epochs are implemented than would typically be expected to be necessary to provide the network with ample of opportunity to fit.

 MODEL TRAINING 
•	The basic approach followed until now is working on the tweets and price (BTC) column as this whole project is about cryptocurrencies. 
•	But changing the approach towards working with sentiment of tweets may give further analysis better.
•	So, the new alternative solution could be building a deep learning model with tweets as X_TRAIN and Sentiment as Y_TRAIN.
•	Sentiment of the tweets can be found by the polarity calculated for each tweet.
•	Sentiment column is created based on the polarity values.
•	If the polarity is >0 the tweet has something positive about the bitcoin. If the polarity <0 then the tweet is negative and the polarity is 0, it says the statement is neutral.
•	Before building the model, Sentiment column is passed to get_dummies so that the words positive, negative, and neutral is converted into 1’s and 0’s format.
•	The same LSTM model is built by considering the tweets as X_TRAIN and Y_TRAIN. Here the model is training itself to see the sentiment of the words present in each tweet.
•	Two LSTM layers are added separately with 256 units each.
•	Activation used: softmax
•	Optimizer: adamax
•	Loss function: mean_squared_error.
•	When using the early stopping, patience=8 is given.
•	Patience is used to see number of epochs without improvement after which training will be early stopped.
•	In this model, the training has completed at 2nd epoch.
•	The accuracy I have got for this approach is 75%, which is quite better improvement compared to the 1st approach. 
IMPROVEMENTS:

•	To start, we will begin by establishing the sequence model. The next step is to add a word embedding layer with 5000 units, a 256-unit LSTM layer, and a fully linked (dense) layer with one (output) neuron that uses relu activation.
•	Mean-squared-error loss and the adamax optimizer are the tools that we make use of when training the model. The accuracy of the model was another one of the metrics that we utilised to measure its performance.
•	The model's summary can be found in the next cell's output.

MODEL SUMMARY OF LSTM 

•	It's also important to point out that we conducted experiments using a stacked LSTM model with two layers. The amount of time spent in training for these models was extended, which resulted in a marked improvement in their overall performance.
•	During the training phase for this model, I have provided a total of one hundred epochs. The model's training was finished during the 51st epoch of time.
•	During training, the monitoring focuses on the accuracy of the validation, and the accuracy of the model is determined by taking the average of the validation accuracy value.
•	During training, the validation accuracy at the 1st epoch is at a level of 64 percent.
By the time the training is completed at the 51st period, the accuracy had reached 89 percent.
•	The argmax () function is used before the accuracy is calculated since it gives the indices of the maximum element of the array for a given axis. This is necessary to calculate the accuracy.
 
MODEL EVALUATION

•	Model evaluation is the process of using different evaluation metrics to understand a model’s performance.
•	Most popular metrics include accuracy, confusion matrix, precision etc.
•	The evaluation is done between the tested data and predicted data which gives the impact of tweets on BTC prices.


