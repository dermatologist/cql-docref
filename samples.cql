define "NeuroSurgeryExample":
    exists (
        [DocumentReference] D
        where D.allergies="Penicillin"
        and D.complaint="Headache"
        and D.complaint="Weakness" or D.complaint="Numbness"
        and D.findings="Intact sensation to light touch"
        )

If the document mentions that the patient has a penicillin allergy, complains of a headache, weakness, or numbness, and has intact sensation to light touch,
then say yes, else say no.


define "DiverticulitisPostOpFeverExample":
    exists (
        [DocumentReference] D
        where D.diagnosis="Diverticulitis"
        and D.complaint="Fever"
        and D.procedure="Colon resection"
        and not D.finding="Fluid collection"
        )

If the document mentions that the patient has diverticulitis, complains of a fever, had a colon resection, and does not have a fluid collection, then say yes, else say no.


define "Test1":
    exists (
        [DocumentReference] D
        where D.complaint="Facial weakness"
        and D.pastMedicalHistory="GERD"
        and D.familyHistory="Stroke"
        and D.finding="gait steady"
        and D.finding="Rhomberg negative"
        )

define "Test2":
    exists (
        [DocumentReference] D
        where D.procedure="Cardiac catheterization"
        and D.diagnosis="Myocardial infarction"
        and D.finding="ST elevations"
        and D.pastMedicalHistory="Spinal stenosis"
        )


define "Test3":
    exists (
        [DocumentReference] D
        where D.complaint="Head ache"
        and D.procedure="craniotomy"
        and D.pastMedicalHistory="Hypertension"
        and D.medication="Dexamethasone"
        )

define "Test4":
    exists (
        [DocumentReference] D
        where D.procedure="Laparoscopy with biopsy"
        where D.pastMedicalHistory="Prostate cancer"
        where D.medication="Docusate Sodium"
        where D.diagnosis="Urethral stricture"
        )