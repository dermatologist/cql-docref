import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import time

questions = pd.read_csv('map/diagnosis_questions.csv')
main_data = pd.read_csv('map/discharge_sample.csv')
subject_ids = questions['subject_id'].unique()

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct",
                                             device_map="auto",
                                            #  attn_implementation="flash_attention_2",
                                            #  torch_dtype=torch.float16
                                             )
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt_templates = [
    (
        "You are a clinician with a patient document. \n"
        "You will be given a document and the concept. You have to answer with a yes or no.\n"
        "Do not include your tasks or instructions.\n"
        "Do not include name, date, or other identifying information.\n"
        "document: {document} \n"
        "{question} yes or no:: "
    ),
]

def stream_data(subject_id):
    docs = []
    for index, notes in main_data[main_data['subject_id'] == subject_id].iterrows():
        document = notes['text']
        question = questions[questions['subject_id'] == subject_id]['question'].values[0]
        print(f"Subject ID: {subject_id}, Question: {question}")
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1280, chunk_overlap=5
        )
        split_docs = text_splitter.split_text(document)
        for doc in split_docs:
            docs.append(prompt_templates[0].format(document=document, question=question))
    for doc in docs:
        yield doc


for subject_id in subject_ids:
    for response in generator(stream_data(subject_id), max_new_tokens=128):
        print(response)
