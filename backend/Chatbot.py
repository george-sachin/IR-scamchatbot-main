from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, Conversation
import os
import pysolr
import requests
import json
import urllib
from urllib.parse import urlencode
import string
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from joblib import load
import itertools
import random
import numpy as np
from strsimpy import Cosine
from chitchat_classifier import ChitChatClassifier
import json
import os

IP = "35.232.124.255"
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

logfile = "./count_logs.json"
print("Initializing logs: ")
if os.path.exists(logfile):
  with open(logfile, "r") as f:
    logs = json.load(f)
else:
  logs = {"chitchat": 0, "healthcare": 0, "environment": 0, "politics": 0, "technology": 0, "education": 0}
  with open(logfile, "w") as f:
    json.dump(logs, f)
print("Logs Initialized!")

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


class Chatbot:
    
  def __init__(self, vectorizer_path = 'model/tf_idf.joblib', qa_model_name = "yjernite/bart_eli5", corp_path='model/tf_idf.csv'):
    self.qa_model, self.qa_tokenizer, self.device = self.initialize_model(qa_model_name)
#     self.conv_model, self.conv_tokenizer, _ = self.initialize_model(conv_model_name)
    self.vectorizer = load(vectorizer_path)
    self.corp = pd.DataFrame(
      data=pd.read_csv('model/tf_idf.csv').values[:,1:],
      index=["healthcare", "environment", "politics", "technology", "education"], 
      columns=self.vectorizer.get_feature_names_out()
    )
    self.user_input = list()
    self.bot_response = list()
    self.chitchat_clf = ChitChatClassifier()
  
  def reset_conv(self):
    self.user_input.clear()
    self.bot_response.clear()

  def initialize_model(self, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    
    return model, tokenizer, device

  def clean_query(self, line):
    punc = string.punctuation.replace("#", "")
    if isinstance(line, list):
        line = line[0]
    query = line.translate(str.maketrans('', '', punc))
    queryText = query.replace("\n","")
    return queryText

  def get_topic(self, cleaned_query):
  #     input_string_clean = clean_string(input_string)
    input_string_vector = self.vectorizer.transform([cleaned_query])
    ind, pair_dist_arg = pairwise_distances_argmin_min(input_string_vector, self.corp, metric='euclidean')
    return self.corp.index[ind[0]]   

  def get_context(self, queryText, topic, num_rows = 5):
    query = {
        "fl" : "* score",
        "q": f"body: {queryText} parent_body: {queryText} selftext: {queryText} topic: {topic}",
        "rows": num_rows,
        "defType": "edismax",
        "wt": "json",
        "qf": "parent_body^0.5 body^0.5 selftext^0.5 topic^2"
    }
    result = urlencode(query)
    inurl = f'http://{IP}:8983/solr/reddit_index_1/select?'+ result

    data = urllib.request.urlopen(inurl).read()
    docs = json.loads(data.decode('utf-8'))['response']['docs']
    df = pd.DataFrame(docs)
    l = []
    for i in range(len(df)):
      if isinstance(df['parent_body'].iloc[i], str):
        l.append(df['parent_body'].iloc[i])
      if isinstance(df['body'].iloc[i], str):
        l.append(df['body'].iloc[i])
      if "selftext" in df.columns:
        if isinstance(df["selftext"].iloc[i], str):
          l.append(df["selftext"].iloc[i])
      

    res = "<P> " + " <P> ".join(l)
    res = res = res.replace("\n","")
    
#     print(f'topic: {topic}, context: {res}')

    return res

  def get_context_scam(self, queryText, num_rows=10):
    query = {
        "fl": "* score",
        "q": f"utterance: {queryText} response: {queryText}",
        "rows": num_rows,
        "defType": "edismax",
        "wt": "json",
        "qf": "response^2 utterance^0.5"
    }
    result = urlencode(query)
    inurl = f'http://{IP}:8983/solr/scam_dataset/select?' + result

    data = urllib.request.urlopen(inurl).read()
    docs = json.loads(data.decode('utf-8'))['response']['docs']
    df = pd.DataFrame(docs)
    l = []
    for i in range(len(df)):
        if "response" in df.columns:
            if isinstance(df["response"].iloc[i], str):
                l.append(df["response"].iloc[i])
            elif isinstance(df["response"].iloc[i], list):
                l.append(df["response"].iloc[i][0])
        # if "utterance" in df.columns:
        #     if isinstance(df["utterance"].iloc[i], str):
        #         l.append(df["utterance"].iloc[i])
        #     elif isinstance(df["utterance"].iloc[i], list):
        #         l.append(df["utterance"].iloc[i][0])

    res = "<P> " + " <P> ".join(l)
    res = res = res.replace("\n", "")

    #     print(f'topic: {topic}, context: {res}')

    return res

  def generate_context_based_response(self, cleaned_query, topic, num_rows = 5):
    context = self.get_context(cleaned_query, topic, num_rows)
    query_and_docs = "question: {} context: {}".format(cleaned_query, context)
    model_input = self.qa_tokenizer(query_and_docs, truncation=True, padding=True, return_tensors="pt")

    generated_answers_encoded = self.qa_model.generate(input_ids=model_input["input_ids"].to(self.device),
                                            attention_mask=model_input["attention_mask"].to(self.device),
                                            min_length=32,
                                            max_length=64,
                                            do_sample=False, 
                                            early_stopping=True,
                                            num_beams=4,
                                            temperature=1.0,
                                            top_k=None,
                                            top_p=None,
                                            eos_token_id=self.qa_tokenizer.eos_token_id,
                                            no_repeat_ngram_size=3,
                                            num_return_sequences=1)
    response = self.qa_tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    return response[0].strip()

  def generate_scam_context_based_response(self, cleaned_query, num_rows=5):
      context = self.get_context_scam(cleaned_query, num_rows)
      query_and_docs = "question: {} context: {}".format(cleaned_query, context)
      model_input = self.qa_tokenizer(query_and_docs, truncation=True, padding=True, return_tensors="pt")

      generated_answers_encoded = self.qa_model.generate(input_ids=model_input["input_ids"].to(self.device),
                                                         attention_mask=model_input["attention_mask"].to(self.device),
                                                         min_length=32,
                                                         max_length=64,
                                                         do_sample=False,
                                                         early_stopping=True,
                                                         num_beams=4,
                                                         temperature=1.0,
                                                         top_k=None,
                                                         top_p=None,
                                                         eos_token_id=self.qa_tokenizer.eos_token_id,
                                                         no_repeat_ngram_size=3,
                                                         num_return_sequences=1)
      response = self.qa_tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
      return response[0].strip()
  
  def generate_conversational_response(self, line, core, num_rows=10):
    queryText = self.clean_query(line)
    query = {
        "fl" : "* score",
        "q": f"utterance: {queryText}",
        "rows": num_rows,
        "defType": "edismax",
        "wt": "json",
        "qf": "utterance^1"
    }
    result = urlencode(query)
    inurl = f'http://{IP}:8983/solr/{core}/select?'+ result

    data = urllib.request.urlopen(inurl).read()
    docs = json.loads(data.decode('utf-8'))['response']['docs']
    df = pd.DataFrame(docs)
    utterances = df['utterance'].to_list()
    responses = df['response'].to_list()
#     print(f"responses: {responses}")
    utt_scores = np.array(list(map(lambda x: cosine_sim(self.clean_query(x), queryText), utterances)))
    res_scores = np.array(list(map(lambda x: cosine_sim(self.clean_query(x), queryText), responses)))
    scores = 0.6*utt_scores+0.4*res_scores
#     print(scores)
    ind = np.argmax(scores)
    response = responses[ind]
    return response
    
#  def generate_conversational_response(self, line):
#     if len(self.user_input)+len(self.bot_response)>self.history_length+1: 
#       utterance = [f"{inp} </s> <s>{res}</s>" for inp,res in zip(self.user_input[-1*self.history_length//2-1:-1], self.bot_response[-1*self.history_length//2:])]
#     else:
#       utterance = [f"{inp} </s> <s>{res}</s>" for inp,res in zip(self.user_input[:-1], self.bot_response[:])]
#     utterance.append(f"<s> {self.user_input[-1]}")
#     print(utterance)
#     model_input = self.conv_tokenizer(utterance, truncation=True, padding=True, return_tensors="pt")
#     generated_responses_encoded = self.conv_model.generate(input_ids=model_input["input_ids"].to(self.device),
#                                             attention_mask=model_input["attention_mask"].to(self.device),
#                                             min_length=16,
#                                             max_length=32,
#                                             do_sample=False, 
#                                             early_stopping=True,
#                                             num_beams=4,
#                                             temperature=1.0,
#                                             top_k=None,
#                                             top_p=None,
#                                             eos_token_id=self.qa_tokenizer.eos_token_id,
#                                             no_repeat_ngram_size=3,
#                                             num_return_sequences=1)
#     response = self.conv_tokenizer.batch_decode(generated_responses_encoded, skip_special_tokens=True,clean_up_tokenization_spaces=True)
#     return response[0]

  def generate_response(self, line, topic = None):
    self.user_input.append(line)
    cls, prob = self.chitchat_clf.predict(line)
#     print(f"class: {cls}, prob: {prob}")
    topic = "scam"
    if topic and topic == "scam":
            context_based = True
            if context_based:
                queryText = self.clean_query(line)
                response = self.generate_scam_context_based_response(queryText)
            else:
                response = self.generate_conversational_response(line, core="scam_dataset")
    else:
        if not topic:
          if cls == "chitchat" and prob<0.535:
            cls = "not-chitchat"
          elif cls == "not-chitchat" and prob<0.6:
            cls = "chitchat"
          print(cls)
          if cls == 'not-chitchat':
            queryText = self.clean_query(line)
            topic = self.get_topic(queryText)
            print(topic)
            logs[topic] += 1
            response = self.generate_context_based_response(queryText, topic)
          else:
            logs["chitchat"] += 1
            response = self.generate_conversational_response(line, core="chitchat_dataset")
        else:
          logs[topic] += 1
          queryText = self.clean_query(line)
          response = self.generate_context_based_response(queryText, topic)
    self.bot_response.append(response)
    with open(logfile, "w") as f:
      json.dump(logs, f)
    return response


if __name__ == "__main__":
    cb = Chatbot()
    # input_string = "Heathcare system is crumbling"
    # input_string = "How is the weather today?"
    # input_string = "I have complaint"
    # response = cb.generate_response(input_string)
    # print(response)
    lst = os.listdir("transcripts/")  # your directory path
    number_files = len(lst)
    with open(f"transcripts/transcript_{number_files}.txt", "a") as file:
        text_transcript = "Chatbot: Hi, I am Tim from Amazon Customer Service. How may I help you?"
        print(text_transcript)
        file.write(text_transcript)
    text = ""
    while True:
        with open(f"transcripts/transcript_{number_files}.txt", "a") as file:
            text_transcript = ""
            raw_text = input(">>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input(">>> ")
            text += raw_text + " "
            text_transcript += "User: " + raw_text + "\n"
            out_text = cb.generate_response(text)
            if isinstance(out_text, list):
                out_text = out_text[0]
            print("Chatbot: ", out_text)
            text_transcript += "Chatbot: " + out_text + "\n"
            file.write(text_transcript)
