from flask import Flask
from flask_restful import Api, Resource, reqparse
import pickle
import os
import sys
import spacy
from nlpkit import preprocessing

app = Flask(__name__)
api = Api(app)

parser_args = reqparse.RequestParser()
parser_args.add_argument('Resume', type=str, help='Resume is required', required=True)

class Parser(Resource):

    def get(self):
        args = parser_args.parse_args()
        resume = args['Resume']

        resume_model = spacy.load('/root/Machine/resume/resume_model')
        doc = resume_model(resume)
        parsed_resume = preprocessing.extract_ents(doc)
     
        return parsed_resume

api.add_resource(Parser, '/parse')

port = int(os.environ.get('PORT', 4444))

if __name__ == "__main__":
    app.run(port=port, debug=True)