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
        and D.pastMedicalHistory="Prostate cancer"
        and D.medication="Docusate Sodium"
        and D.diagnosis="Urethral stricture"
        )

define "Test5":
    exists (
        [DocumentReference] D
        where D.diagnosis="rheumatoid arthritis"
        and D.finding="thickened pericardium"
        and D.pastMedicalHistory="Dyslipidemia"
        and D.medication="Colchicine"
        )

define "Test6":
    exists (
        [DocumentReference] D
        where D.complaint="fever"
        and D.finding="obesity"
        and D.medication="Ativan"
        and D.pastMedicalHistory="Thyroid nodule"
        and D.imagingObservation="Mild pulmonary edema"
        )

define "Test7":
    exists (
        [DocumentReference] D
        where D.complaint="slurred speech"
        and D.pastMedicalHistory="Atrial fibrillation"
        and D.medication="atorvastatin"
        and not D.imagingObservation="evidence of stroke"
        )

define: "Test8":
    exists (
        [DocumentReference] D
        where D.allergyIntolerance="Sulfonamide Antibiotics"
        and D.complaint="urinary retention"
        and D.procedure="posterior colporrhaphy"
        and D.pastMedicalHistory="Asthma"
        )

define "Test9":
    exists (
        [DocumentReference] D
        where D.allergyIntolerance="Penicillins"
        and D.complaint="fall"
        and D.medication="warfarin"
        and D.diagnosis="C2 fracture"
        )

define "Test10":
    exists (
        [DocumentReference] D
        where D.complaint="suicidal"
        and D.pastMedicalHistory="Depression"
        and D.familyHistory="hypothyroidism"
        and D.medication="Clonazepam"
        )

define "Test11":
    exists (
        [DocumentReference] D
        where D.complaint="Epistaxis"
        and D.finding="hypertension"
        and D.pastMedicalHistory="Hypercholesterolemia"
        and D.medication="Clopidogrel"
        )