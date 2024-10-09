from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2)
llm = HuggingFacePipeline(pipeline=pipe)

template = """
You are an assistant that can say if a concept is present in a document or not.
You will be given a document and the concept. You have to answer with a yes or no.
document: {document}
concept: {question}
Is the concept present in the document? yes or no: """

prompt = PromptTemplate.from_template(template)






# print(output_parser.guard.base_prompt)
# prompt = PromptTemplate(
#             template=output_parser.get_prompts(),
#             input_variables=["document", "question"],
#         )

chunk = {
    "document": "Electroencephalography (EEG) is an electrophysiological monitoring method to record electrical activity of the brain. It is typically noninvasive, with the electrodes placed along the scalp.",
    "question": "brain"
}

chain = prompt | llm | StrOutputParser()
answer = chain.invoke(chunk).split(" ")[-1].lower().strip()
print(answer)
# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)

# chain = prompt | hf

# question = "What is electroencephalography?"

# print(chain.invoke({"question": question}))