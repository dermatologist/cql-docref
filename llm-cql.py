from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
import pandas as pd
import tqdm
import torch

model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=16)
llm = HuggingFacePipeline(pipeline=pipe)

assertions = pd.read_csv('map/diagnosis.csv')


MAP_TEMPLATE = """
You are an assistant that can convert a CQL query to a natural language.
You should give a single line answer>> as in the example below.
Example:

CQL: exists (["DocumentReference": "Diabetes Mellitus"])
answer>> Does the document ascertain Diabetes Mellitus?
CQL: exists (["DocumentReference": "Herpes Zoster"])
answer>> Does the document ascertain Herpes Zoster?

Now convert the following CQL query to a natural language.
CQL: exists (["DocumentReference": {diagnosis}])

answer>> """

map_prompt = PromptTemplate.from_template(MAP_TEMPLATE)

ASSERT_TEMPLATE = """
You will be given a document and the concept. You have to answer with a yes or no.
document: {document}
{question} yes or no: """

assert_prompt = PromptTemplate.from_template(ASSERT_TEMPLATE)


# data = []
# for i, row in assertions.iterrows():
#     chunk = {
#         "diagnosis": row["diagnosis"]
#     }
#     chain = map_prompt | llm | StrOutputParser()
#     response = chain.invoke(chunk)
#     question = response.split(">>")[-1].strip()
#     data.append([row['subject_id'], row['diagnosis'], question])
#     print(question)
# _df = pd.DataFrame(data, columns=['subject_id', 'diagnosis', 'question'])
# _df.to_csv('map/diagnosis_questions.csv', index=False)


questions = pd.read_csv('map/diagnosis_questions.csv')
main_data = pd.read_csv('map/discharge_sample.csv')

def chunk_notes(subject_id):
    docs = []
    for index, notes in main_data[main_data['subject_id'] == subject_id].iterrows():
        discharge_note = notes['text']
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=5
        )
        split_docs = text_splitter.split_text(discharge_note)
        for doc in split_docs:
            docs.append(doc)
    return docs

def assert_diagnosis(docs, question):
    torch.cuda.empty_cache()
    for doc in tqdm.tqdm(docs):
        chunk = {
            "document": doc,
            "question": question
        }
        chain = assert_prompt | llm | StrOutputParser()
        answer = chain.invoke(chunk).split(" ")[-1].lower().strip()
        if "yes" in answer:
            return True
    return False

for i, row in questions.iterrows():
    subject_id = row['subject_id']
    question = row['question']
    docs = chunk_notes(subject_id)
    print(f"Subject ID: {subject_id}, Question: {question}")
    print(f"Answer: {assert_diagnosis(docs, question)}")

