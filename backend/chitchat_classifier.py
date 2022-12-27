from pickle import load
from re import sub

def preprocess(excerpt):
    """ Return a cleaned up excerpt.
    """
    excerpt = sub('[^a-zA-Z]', ' ', excerpt)
    excerpt = excerpt.lower()
    excerpt = excerpt.split()
    excerpt = ' '.join(excerpt)
    return excerpt


def document_term_matrix(vectorizer, excerpt):
    """ Return a document-term matrix.
    """
    return vectorizer.transform([preprocess(excerpt)]).toarray()

class ChitChatClassifier:
    def __init__(self):
        with open('model/chitchat1.model', 'rb') as f:
            self.model =  load(f)
        with open('model/tfidf1.vectorizer', 'rb') as f:
            self.vec = load(f)
        self.labels = ('not-chitchat', 'chitchat')

    def predict(self, excerpt):
        x = document_term_matrix(self.vec, excerpt)
        p = self.model.predict_proba(x)[0]
        d = dict(zip(self.model.classes_, p))
        key = max(d, key=d.get)
        return self.labels[key], d[key]