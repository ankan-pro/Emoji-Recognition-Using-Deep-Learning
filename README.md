# Emoji-Recognition-Using-Deep-Learning

Abstract
									 
Emojis are an inevitable data emerging across the last years, from marketing, digital communication in particular. Emojis helps individuals to express feelings, emotions, and thoughts during text conversations. As the use of social media is increased, the usage of emojis also increased drastically. There is various numbers of emoji prediction techniques but prediction depending on user’s speech or text recommendation still persists. This project proposes how to predict an emoji based on given text or phrase. That means here we build a text classifier that returns an emoji that suits the given text. Our methodology consist Exploratory Data Analysis, Build the classifier model and Train and evaluate the model. Our approach differs from existing studies and improves the accuracy of emoji prediction.

 





 
Introduction

Emojis are becoming a whole new language that can more effectively express an idea or emotion. This visual language is now a standard for online communication, available not only in Twitter, but also in another large online platform such as Facebook and Instagram. So we worked on a project where we tried to predict emojis based on users recommendation. We used, Recurrent Neural Networks (RNN) with Long-Term Short Memory (LSTM) deep learning algorithm to build this model. A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes can create a cycle, allowing output from some nodes to affect subsequent input to the same nodes. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition. Recurrent neural networks are theoretically Turing complete and can run arbitrary programs to process arbitrary sequences of inputs.
The term "recurrent neural network" is used to refer to the class of networks with an infinite impulse response, whereas "convolutional neural network" refers to the class of finite impulse response. Both classes of networks exhibit temporal dynamic behavior. A finite impulse recurrent network is a directed acyclic graph that can be unrolled and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a directed cyclic graph that cannot be unrolled. Long short-term memory (LSTM) is an artificial neural network used in the fields of artificial intelligence and deep learning. Unlike standard feed forward neural networks, LSTM has feedback connections. Such a recurrent neural network (RNN) can process not only single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition, machine translation, robot control, video games, and healthcare. LSTM has become the most cited neural network of the 20th century. This model is effective in various other closely related tasks, including sentiment classification and language modeling.

 
Objective

We are doing this project so that we can create a model for predicting emoji from custom user input which is fast and uses LSTM and RNN with high accuracy. The goal of this project is to predict an emoji that is associated with a text message. To accomplish this task, we train and test several neural network models on a data to predict a sentiment associated with a text message. Then, we represent the predicted sentiment as an emoji. Our project takes input directly from the user unlike other models present. We have different classes of emoji that will check from our training database and give an acute result. Our project has a high accuracy and which gives result at a very fast time. Thus, our attempt at improving already existing models.
 


Survey:


Prior studies have focused on extracting meaningful data from text inputs for emoji prediction. Some researchers, such as [4], explore the usage of labels for emoji prediction, wherein they assign a value to essential words in sentences through an attention mechanism. Bi-LSTM (Bidirectional Long Short-Term Memory), Fast Text and Deep emoji models assign labels and weights to different parts of the text for
estimating probabilities used for emoji prediction. Some studies use neural networks for emoji prediction on a Twitter dataset [12]. LSTM (Long Short-Term Memory) and CNN (Convolutional Neural Networks) outperform models in the comparative studies. Although these models provide promising results, there is a need for identifying more heuristic approaches.


For further advancement in emoji prediction, many papers explore more parameters to predict emoji, in addition to text. [5] focus on categorizing emoji labels using text and image data to train semantic embeddings. The authors conduct a comparative study
to compare predictions using only text input, visual data input, and its fusion. The results of this hybrid model indicate that multiple features and text can help in efficient emoji prediction.


Similarly, [1] combine visual and textual information for Instagram posts. The images uploaded in the posts give visual information, and the captions give textual information. Their result  sindicate that combining these two synergistic approaches in a single model improves the accuracy.


More features to predict emojis are presented by[2]. The authors conduct a study on the variation of emoji usage across different seasons. The paper uses a Twitter datasetwith four subsets depending on the season at the time of posting. The four divisions are Spring, Summer, Autumn, and Winter. The LSTM model highlights the dependence of some emojis on the time of year. [11] propose a fusion of contextual and personal
features. Contextual features from the text message and additional features such as user preference, gender, and current time give emoji predictions in a more personalized manner using a score-ranking matrix factorization framework.


Some studies [10] focus on multi-turn dialogue systems for emoji prediction, and the dataset used is the Weibo2015 dialogues in Chinese. A novel model called H-LSTM (Hierarchical Long Short-Term Memory) is used to extract the meaning of sentences and then predict a suitable emoji for the reply. Hence, the authors demonstrate that predicting an emoji based on contextual information extracted from the messages gives better results.[9][11] focus on predicting emojis influenced by user personality during message conversations. Test scenarios presented to users in a survey helped to get their
preferences of emojis. The results show that personality features impact the emoji predicted.


Previous research work [12] uses datasets from online social media platforms for emoji use. However, it does not consider the conversational aspect in datasets and the usage dependency on the sentiment of message, exact word mapping, and semantic similarity.
Emojis are more prevalent on messaging platforms, and to the best of our knowledge, no study has focused on using conversational data for emoji prediction. Hence, we propose an approach that addresses the conversational dataset issue. We use the time and location of text messages in predicting emojis as these impact the tone of the message.
They help in emoji prediction by extracting the sentiment of the text message.

 
Drawbacks

All the models which exist online for the emoji prediction, take a separate test dataset, our model takes custom input for emoji prediction in real time from user. The previous projects cannot decipher noisy data. The other models don’t have high accuracy either. Our project has a very high accuracy and can deal with noisy data too. We have trained and embedded our model with many data from the dataset. This helps us deal with noisy data and gives us proper high accuracy outputs. We also have implemented padding with the code, which helps us give a proper emoji as prediction despite the length of the sentence.
 
 
Tools and Libraries:

•	Python – 3.x   
                     
•	Numpy – 1.19.2
                      
•	Pandas – 1.2.4
                          
•	TensorFlow – 2.4.x
                     
•	Emoji – 1.2.0
                                                   
 
Methodology:
Collection of Data:
This project paper presents datasets consisting of most frequently used conversational texts and phrases.	We go through the internet, various social media to collect these data. Along with the text or phrases the database consist a numeric column where emoji no is stored. Also we used emoji data base.
 
 
Perform Exploratory Data Analysis:
First we imported important modules like pandas,emoji,numpy,keras,sequencial,embeddings ,tokenizer etc
import numpy as np
import pandas as pd
import emoji

from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Embedding
from keras_preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

Then we read the training dataset
import pandas as pd
data = pd.read_csv('/content/train_emoji.csv',header=None)
data.head()
 
Let’s store the above information in a dictionary for ease of use
emoji_dict = {
    0: ":red_heart:",
    1: ":baseball:",
    2: ":grinning_face_with_big_eyes:",
    3: ":disappointed_face:",
    4: ":fork_and_knife_with_plate:"
}

def label_to_emoji(label):
    return emoji.emojize(emoji_dict[label])

Now using the emoji module, see how these emojis turn out
import emoji
for ix in emoji_dict.keys():
    print (ix,end=" ")
    print (emoji.emojize(emoji_dict[ix]))
 
EMBEDDINGS
We use word embeddings in this emoji prediction project to represent the text.
The relationship between the words is represented using word embeddings. This process aims to create a vector with lesser dimensions. An embedding is a low-dimensional space into which high-dimensional vectors can be translated. Machine learning on huge inputs, such as sparse vectors representing words, is made simpler via embeddings.
We use the 6B 50D GloVe vector to build the embedding matrix for the text in our dataset
file =open('/content/glove.6B.50d.txt','r',encoding ='utf8')
content=file.readlines()
file.close()

embeddings = {}

for line in content:
    line = line.split()
    embeddings[line[0]] = np.array(line[1:], dtype = float)


 
Now we create the input sentences into tokens 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
word2index = tokenizer.word_index
Find the maximum length of the  data
def get_maxlen(data):
    maxlen = 0
    for sent in data:
        maxlen = max(maxlen, len(sent))
    return maxlen

maxlen = get_maxlen(Xtokens)
print(maxlen)

Thereafter we padded the data sequences
Xtokens=tokenizer.texts_to_sequences(X)
Xtrain = pad_sequences(Xtokens, maxlen = maxlen,  padding = 'post', truncating = 'post')


Xtrain

 
Ytrain = to_categorical(Y)


.

embed_size = 50
embedding_matrix = np.zeros((len(word2index)+1, embed_size))
for word, i in word2index.items():
    embed_vector = embeddings[word]
    embedding_matrix[i] = embed_vector

embedding_matrix




 











Train and evaluate the model :
For this emoji prediction project, we will be using a simple LSTM network.
LSTM stands for Long Short Term Network. Recurrent neural networks are a type of deep neural network used to deal with sequential types of data like audio files, text data, etc.
LSTMs are a variant of Recurrent neural networks that are capable of learning long-term dependencies. LSTM networks work well with time-series data.
Let’s build a simple LSTM network using TensorFlow.


model = Sequential([
    Embedding(input_dim = len(word2index) + 1,
              output_dim = embed_size,
              input_length = maxlen,
              weights = [embedding_matrix],
              trainable = False
             ),
    
    LSTM(units = 16, return_sequences = True),
    LSTM(units = 4),
    Dense(5, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

 
Train the model :
model.fit(Xtrain, Ytrain, epochs = 50)

 

 
 








Results and Discussion:
In this deep learning project, we built a text classifier that predicts an emoji that suits the given text. We achieve good accuracy in the implementation, although based on requirements we can train it with larger datasets. To predict emojis, we used LSTM as LSTM networks work well with series data.

Let’s predict the emoji labels on the testing data.
test = ["lets play", "I feel very bad", "lets eat dinner"]

test_seq = tokenizer.texts_to_sequences(test)
Xtest = pad_sequences(test_seq, maxlen = maxlen, padding = 'post', truncating = 'post')

y_pred = model.predict(Xtest)
y_pred = np.argmax(y_pred, axis = 1)

for i in range(len(test)):
    print(test[i], label_to_emoji(y_pred[i]))



Output:

 


Future Scopes:

In near future the more training data-bases with most occurring sentences will build and trained with this model the accuracy will increase even more .In the social media apps it will help people to find suitable emojis corresponding to their written phrase or texts.

 

Conclusion:

Although our initial work is promising, the models investigated still have significant room for improvement. The major part of the difficulty in this task is working with a noisy dataset. Message texts can be quite unstructured, especially with emojis being used much more loosely than real words. Different people may have different kind of views related to their message. So cleaning up most of noisy data will helps us in classification performance.








References:
1.	Barbieri F, Ballesteros M, Ronzano F, Saggion H (2018) Multimodal Emoji Prediction
2.	Barbieri F, Ballesteros M, Ronzano F, Saggion H (2018) Multimodal Emoji Prediction. CoRR abs/1803.02392
3.	Barbieri F, Ballesteros M, Saggion H (2017) Are Emojis Predictable? CoRR abs/1702.07285
4.	Barbieri F, Espinosa-Anke L, Camacho-Collados J, Schockaert S, Saggion H (2018) Interpretable Emoji Prediction via Label-Wise Attention LSTMs. pp 4766–4771
5.	Cappallo S, Mensink T, Snoek CGM (2015) Image2Emoji: Zero-Shot Emoji Prediction for Visual Media. In: Proceedings of the 23rd ACM International Conference on Multimedia. Association for Computing Machinery, New York, NY, USA, pp 1311–1314
6.	Devlin J, Chang M-W, Lee K, Toutanova K (2018) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR abs/1810.04805
7.	Hochreiter S, Schmidhuber J (1997) Long Short-Term Memory. Neural Comput 9:1735– 1780. doi: 10.1162/neco.1997.9.8.1735
8.	Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I (2017) Attention is All you Need. ArXiv abs/1706.03762
9.	Völkel ST, Buschek D, Pranjic J, Hussmann H (2019) Understanding Emoji Interpretation through User Personality and Message Context. In: Proceedings of the 21st International Conference on Human-Computer Interaction with Mobile Devices and Services. Association for Computing Machinery, New York, NY, USA
10.	Xie R, Liu Z, Yan R, Sun M (2016) Neural Emoji Recommendation in Dialogue Systems. CoRR abs/1612.04609
11.	Zhao G, Liu Z, Chao Y, Qian X (2020) CAPER: Context-Aware Personalized Emoji Recommendation. IEEE Transactions on Knowledge and Data Engineering PP:1. doi: 10.1109/TKDE.2020.2966971
12.	Zhao L, Zeng C (2017) Using Neural Networks to Predict Emoji Usage
13.	Find Open Datasets and Machine Learning Projects | Kaggle
