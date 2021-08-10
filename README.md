# Resume-Parser-Ranker

HR need to go through a lot of resume in order to select the best candidate for a job. This is time consuming and tiresome.

Resume parsing, also known as CV parsing, resume extraction or CV
extraction, allows for the automated storage and analysis of resume data.
Resume ranking automatically ranks a list of resumes based on the similarity
to the job description. The resume is imported into the software and the
information is extracted and save in a json format. RESParse analyzes a
resume, extract the desired information and stores it in a json format. Once
the resumes have been have been analyzed and the information have been
extracted out, it is the job of the ranker which uses cosine similarity to
measure the similarity between the job description and the resume. A
similarity score of 1 indicates perfect similarity and a similarity score of
0 indicates dissimilarity.

The project works with only Microsoft word files. For testing purposes, I have included a resume in `Data/Obiora-TechCv.docx`, you can also test the algorithm on your personal resume

## Data

Data was collected from kaggle and annotated manually. The data was then
converted to a json file

### Technologies Used

- Jupyter Notebooks
- Python 3.7
- Pandas
- Numpy
- Scikit-Learn
- spaCy
- Html and Css
- Flask

### How to use the Program

To run the application you need to install the modules in `requirements.txt`, if you have them installed already, you can skip step 1

STEP 1: run `pip install requirements.txt`

STEP 2: Open up a terminal and run `api.py`

STEP 3: Navigate to `webapp` directory and run `app.py`

A sample job description can be found in `webapp/job_descr.txt`
