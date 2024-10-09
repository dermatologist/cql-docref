from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64)
llm = HuggingFacePipeline(pipeline=pipe)

assertions = pd.read_csv('map/diagnosis.csv')

MAP_TEMPLATE = """
You are an assistant that can map a CQL query to a natural language question.
You will be given a CQL query. You have to answer with a natural language question as in the example below.
Example:

CQL Query: exists (["DocumentReference": "Diabetes Mellitus"])
Natural Language Question: Does the document ascertain Diabetes Mellitus?

CQL Query: exists (["DocumentReference": {diagnosis}])
Natural Language Question:: """

map_prompt = PromptTemplate.from_template(MAP_TEMPLATE)

for i, row in assertions.iterrows():
    chunk = {
        "diagnosis": row["diagnosis"]
    }
    chain = map_prompt | llm | StrOutputParser()
    response = chain.invoke(chunk)
    question = response.split("::")[-1].strip()
    print(question)