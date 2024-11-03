from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import tqdm
import torch
import os
import time

start_time = time.time()

# Load the model
model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
llm = HuggingFacePipeline(pipeline=pipe)

assertions = pd.read_csv('~/data/diagnosis.csv')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# Define the prompt templates
MAP_TEMPLATE = """
You are an assistant that can convert a CQL query to a natural language.
You should give a single line answer>> as in the example below.
Example:

CQL:     exists (
        [DocumentReference] D
        where D.allergies="Penicillin"
        and D.complaint="Headache"
        and D.complaint="Weakness" or D.complaint="Numbness"
        and D.findings="Intact sensation to light touch"
        )

answer>> Penicillin allergy, Headache, Weakness, Numbness, and Intact sensation to light touch as findings?

CQL:    exists (
            [DocumentReference] D
            where D.diagnosis="Diverticulitis"
            and D.complaint="Fever"
            and D.procedure="Colon resection"
            and not D.finding="Fluid collection"
        )

answer>> Diverticulitis diagnosis, Fever complaint, Colon resection procedure, and no Fluid collection finding?

Now convert the following CQL query to a natural language.
CQL: {cql}

answer>> """

map_prompt = PromptTemplate.from_template(MAP_TEMPLATE)

ASSERT_TEMPLATE = """
You will be given a document and a question.\n
Summarize the document chunk commenting on: {question} \n
Do not include absent or negative mentions.\n
document: {document} \n
Summary:: """

assert_prompt = PromptTemplate.from_template(ASSERT_TEMPLATE)

FINAL_TEMPLATE = """
You will be given a document and a question to answer with ONLY a yes or no.\n
Do not include the document in the answer.\n
Say yes if the document mentions all aspects of the question, else say no.\n
document: {document} \n
question: Does the document mention {question} Say yes or no:: """

final_prompt = PromptTemplate.from_template(FINAL_TEMPLATE)

# Generate the natural language questions from the CQL queries
data = []
for i, row in assertions.iterrows():
    chunk = {
        "cql": row["cql"]
    }
    chain = map_prompt | llm | StrOutputParser()
    response = chain.invoke(chunk)
    question = response.split(">>")[-1].strip()
    data.append([row['subject_id'], row['cql'], question])

_df = pd.DataFrame(data, columns=['subject_id', 'cql', 'question'])
_df.to_csv('~/data/diagnosis_questions.csv', index=False)



def chunk_notes(data):
    docs = []
    for index, notes in data.iterrows():
        discharge_note = notes['text']
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1024, chunk_overlap=5
        )
        split_docs = text_splitter.split_text(discharge_note)
        for doc in split_docs:
            docs.append(doc)
    return docs

# Map
def collect_facts(docs, question):
    facts = ""
    for doc in tqdm.tqdm(docs):
        chunk = {
            "document": doc,
            "question": question
        }
        chain = assert_prompt | llm | StrOutputParser()
        answer = chain.invoke(chunk).split("::")[-1].strip()
        facts += answer + " "
    return facts

# Reduce
def final_answer(facts, question):
    chunk = {
        "document": facts,
        "question": question
    }
    chain = final_prompt | llm | StrOutputParser()
    answer = chain.invoke(chunk).split("::")[-1].strip()
    print(f"Answer: {answer}")
    if "yes" in answer.lower():
        return True
    return False

questions = pd.read_csv('~/data/diagnosis_questions.csv')
main_data = pd.read_csv('~/data/discharge_sample.csv')
previous_answer = False
TP = 0
FP = 0
TN = 0
FN = 0
for i, row in questions.iterrows():
    subject_id = row['subject_id']
    question = row['question']
    data = main_data[main_data['subject_id'] == subject_id]
    docs = chunk_notes(data)
    if len(docs) > 20:
        continue
    facts = collect_facts(docs, question)
    answer = final_answer(facts, question)
    print(f"Subject ID: {subject_id}, Question: {question}, Answer: {answer}, Previous Answer: {previous_answer}")
    # True and false are in sequence
    if not previous_answer and answer:
        TP += 1
    if previous_answer and not answer:
        TN += 1
    if not previous_answer and not answer:
        FN += 1
    if previous_answer and answer:
        FP += 1
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    previous_answer = not previous_answer
print(f"FINAL: TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
print(f"Accuracy: {(TP + TN) / (TP + TN + FP + FN)}")
print(f"Time: {time.time() - start_time}")
