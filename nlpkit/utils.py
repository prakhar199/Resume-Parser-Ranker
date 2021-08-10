# utils.py

import random
import docx 
import spacy

def read_document(path: str) -> str:
    """
    Reads a Microsoft Office document

    Parameters
    ----------
    path: str
        The absolute path of the document 

    Returns
    -------
    str: returns a string object
    """

    text = ''
    doc = docx.Document(path)
    all_paragraphs = doc.paragraphs

    for paragraph in all_paragraphs:
        text = text + paragraph.text + '\n'

    return text

def train_model(nlp, train_data):
    """
    Train a named entity recognition model.

    Parameters
    ----------
    nlp: object
        A blank nlp model

    train_data: list
        A list of tuples, this should be in the format below

        train_data = [
            ("Govardhana K Senior Software Engineer  Bengaluru, Karnataka, Karnataka - Email me on Indeed: indeed.com/r/Govardhana-K/ b2de315d95905b68  Total IT experience 5 Years 6 Months Cloud Lending Solutions INC 4 Month • Salesforce Developer Oracle 5 Years 2 Month • Core Java Developer Languages Core Java, Go Lang Oracle PL-SQL programming, Sales Force Developer with APEX....".
            {'entities': [(1749, 1755, 'Companies worked at'), (1696, 1702, 'Companies worked at'), (1417, 1423, 'Companies worked at'), (1356, 1793, 'Skills'), (1209, 1215, 'Companies worked at'), (1136, 1248, 'Skills'), (928, 932, 'Graduation Year')]....)
        ]

    Returns
    -------
    None: returns None object
    """

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
        
    for _, ents in train_data:
        for ent in ents['entities']:
            ner.add_label(ent[2])
            
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        max_iterations = 20
        print('Running {} iterations'.format(max_iterations))
        print('_____________________________________________')
        for itr in range(max_iterations):
            print('Starting iteration ', str(itr))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, ents in train_data:
                try:
                    nlp.update(
                        [text],
                        [ents],
                        drop=.2,
                        sgd=optimizer,
                        losses=losses,
                    )
                except Exception  as e:
                    pass
                
            print(losses)