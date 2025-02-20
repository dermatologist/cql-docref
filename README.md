# LLM-in-the-loop Execution of Clinical Quality Language (CQL) ([CQL](https://cql.hl7.org/)) on Unstructured Data

LLM-CQL.py is a proof of concept for LLM-in-the-loop execution of [CQL](https://cql.hl7.org/) on unstructured data. We propose using LLMs to execute CQL statements referring FHIR *DocumentReferences* to first, convert CQL to natural language, then map the referenced *DocumentReference* to facts and finally reduce it to a binary label. A fork of the **CQL Execution framework** that provides a hook for integrating LLM-in-the-loop execution of CQL on unstructured data is [here](https://github.com/dermatologist/cql-execution). An end-to-end demo with a CQL execution engine will be released soon!

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Data Preparation

## Sample record (discharge_sample.csv)
note_id | subject_id | hadm_id | note_type | note_seq | charttime | storetime | text |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | XXXXXX5 | 123456 | DS | 1 | 2021-01-01 00:00:00 | 2021-01-01 00:00:00 | Sample Text

### CQL and actual status (diagnosis.csv)

| subject_id | cql | actual status |
| --- | --- | --- |
| XXXXXX5 | exists ([DocumentReference] D where D.procedure="Cardiac catheterization" and D.diagnosis="Myocardial infarction" and D.finding="ST elevations" and D.pastMedicalHistory="Spinal stenosis") | TRUE
| XXXXXX5 | exists ([DocumentReference] D where D.complaint="Facial weakness" and D.pastMedicalHistory="GERD" and D.familyHistory="Stroke" and D.finding="gait steady" and D.finding="Rhomberg negative") | FALSE

### Natural Language Query from CQL (diagnosis_questions.csv)
* *This file is generated by the script in STEP (a).*

| subject_id | cql | Natural Language Query |
| --- | --- | --- |
| XXXXXX5 | exists ([DocumentReference] D where D.procedure="Cardiac catheterization" and D.diagnosis="Myocardial infarction" and D.finding="ST elevations" and D.pastMedicalHistory="Spinal stenosis") | Myocardial infarction diagnosis, Cardiac catheterization procedure, ST elevations finding, and Spinal stenosis past medical history?
| XXXXXX5 | exists ([DocumentReference] D where D.complaint="Facial weakness" and D.pastMedicalHistory="GERD" and D.familyHistory="Stroke" and D.finding="gait steady" and D.finding="Rhomberg negative") | Facial weakness complaint, GERD past medical history, Stroke family history, gait steady finding, and Rhomberg negative finding?






## Usage

To use LLM-CQL.py, run the following command:

```bash
python llm-cql.py
```


## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please open an issue on GitHub.
