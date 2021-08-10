import requests
from nlpkit.utils import read_document

BASE = 'http://127.0.0.1:5000/'

data = {'Resume': read_document("Data/Obiora-TechCv.docx")}

response = requests.get(BASE + 'parse', data)
print(response.json())
