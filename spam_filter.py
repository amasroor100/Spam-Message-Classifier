import streamlit as st
import pickle
import sklearn
import re

from PIL import Image

st.title('Spam Message Classifier')

image = Image.open('Spam.png')
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(image, caption='SPAM ALERT')

training_set_clean = pickle.load(open('words.pkl', 'rb'))
vocabulary = pickle.load(open('vocab.pkl', 'rb'))


spam_message = training_set_clean[training_set_clean["Label"] == "spam"]
ham_message = training_set_clean[training_set_clean["Label"] == "ham"]

p_spam = len(spam_message)/len(training_set_clean)
p_ham = len(ham_message)/len(training_set_clean)

n_word_per_spam_message = spam_message["SMS"].apply(len)
n_spam = n_word_per_spam_message.sum()

n_word_per_ham_message = ham_message["SMS"].apply(len)
n_ham = n_word_per_ham_message.sum()

n_vocabulary = len(vocabulary)

alpha = 1
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}


for word in vocabulary:
    n_word_given_spam = spam_message[word].sum()
    p_word_given_spam = (n_word_given_spam + alpha)/(n_spam + alpha * n_vocabulary)
    parameters_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham_message[word].sum()
    p_word_given_ham = (n_word_given_ham + alpha)/(n_ham + alpha * n_vocabulary)
    parameters_ham[word] = p_word_given_ham



def classify(message):
    """
    message: a string
    """
    
    message = re.sub("\W"," ", message)
    message = message.lower().split()
    
    p_spam_given_messeage = p_spam
    p_ham_given_messeage = p_ham
    
    for word in message:
        if word in parameters_spam:
            p_spam_given_messeage *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_messeage *= parameters_ham[word]
            
    if p_ham_given_messeage > p_spam_given_messeage:
        return "ham"
    elif p_ham_given_messeage < p_spam_given_messeage:
        return "spam"
    else:
        return "needs human classification"
        




input_sms = st.text_input('Enter the Message ')

option = st.selectbox("You Got Message From:-", ["Via SMS ", "Via EMAIL", "Other Source"])


if st.button('Click to Predict'):
    result = classify(input_sms)


    if result == "spam":
        st.header("Spam Message")
    elif result == "ham":
        st.header('Ham Message')
    else:
        st.header("Needs Human to Classify it")


