import gradio as gr
from model import SmokerModel
import numpy as np
import pandas as pd

MODEL = SmokerModel("ensemble_softvoting_model.joblib","min_max_scaler.joblib")

def predict(
        age, height, weight, 
        waist, eye_L, eye_R, 
        hear_L, hear_R, systolic, 
        relaxation, fasting_blood_sugar, cholesterol, 
        triglyceride, HDL, LDL, 
        hemoglobin, urine_protein, 
        serum_creatinine, AST, ALT, 
        Gtp, dental_caries
    ):
    '''
    Predict the label for the data inputed
    '''

    # Create a dictionary with input data and dataset var names
    input_data = {
        "age": age,
        "height(cm)": height,
        "weight(kg)": weight,
        "waist(cm)": waist,
        "eyesight(left)": eye_L,
        "eyesight(right)": eye_R,
        "hearing(left)": hear_L,
        "hearing(right)": hear_R,
        "systolic": systolic,
        "relaxation": relaxation,
        "fasting blood sugar": fasting_blood_sugar,
        "Cholesterol": cholesterol,
        "triglyceride": triglyceride,
        "HDL": HDL,
        "LDL": LDL,
        "hemoglobin": hemoglobin,
        "Urine protein": urine_protein,
        "serum creatinine": serum_creatinine,
        "AST": AST,
        "ALT": ALT,
        "Gtp": Gtp,
        "dental caries": dental_caries
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    #predict
    label = MODEL.predict(input_df)

    return label

def load_examples(csv_file):
    '''
    Load examples from csv file
    '''
    # Read examples from CSV file
    df = pd.read_csv(csv_file)

    # Convert DataFrame to a list of lists
    examples = df.values.tolist()

    return examples

def load_interface(): 
    '''
    Configure Gradio interface
    '''

    #set blocks
    info_page = gr.Blocks()

    with info_page:
        # set title and description
        gr.Markdown(
        """
        # Ensemble Classifier for Predicting Smoker or Non-Smoker
        
        **Contributors**: Matt Soria, Jake Leniart, Francisco Lozano\n
        **University**: Depaul University\n
        **Class**: DSC 478, Programming Machine Learning\n

        ## Overview
        Our project focused on creating a classifier for a Kaggle dataset containing bio-signals and information on individuals' smoking status. The classifier aims to identify whether a patient is a smoker based on 22 provided features. You can find the dataset [here](https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals?resource=download&select=train_dataset.csv). 
        We developed an Ensemble Classifier with Soft Voting, which combines KNN, SVM, and XGBoost classifiers.

        ## Labels
        - **non-smoker** = 0 
        - **smoker** = 1

        ## Classifier Metrics

        ### Classification Report

        ```
        Train Accuracy:  0.7833977837414656
        Test Accuracy:   0.7885084006669232

                      precision    recall  f1-score   support

          non-smoker       0.83      0.84      0.83      4933
              smoker       0.72      0.69      0.71      2864

            accuracy                           0.79      7797
           macro avg       0.77      0.77      0.77      7797
        weighted avg       0.79      0.79      0.79      7797
        ```

        ## Confusion Matrix

        ![](file/smoker_cm.png)

        ## Final Report
        For more details about our Ensemble Classifier and the individual models, please refer to our Jupyter notebooks in our project repository.\n
        [DSC 478 Project Repo](https://github.com/FranciscoLozCoding/smoker_classifier)
        """
        )

    age = gr.Number(label="Age", precision=0, minimum=0)
    height = gr.Number(label="Height(cm)", precision=0, minimum=0)
    weight = gr.Number(label="Weight(kg)", precision=0, minimum=0)
    waist = gr.Number(label="Waist(cm)", minimum=0, info="Waist circumference length")
    eye_L = gr.Number(label="Visual acuity of the left eye, measured in diopters (D)", minimum=0)
    eye_R = gr.Number(label="Visual acuity of the right eye, measured in diopters (D)", minimum=0)
    hear_L = gr.Radio(label="Is there any hearing ability in the left ear?",choices=[("Yes",1),("No",2)])
    hear_R = gr.Radio(label="Is there any hearing ability in the right ear?",choices=[("Yes",1),("No",2)])
    systolic = gr.Number(label="Systolic(mmHg)", precision=0, minimum=0, info="Blood Pressure")
    relaxation = gr.Number(label="Relaxation(mmHg)", precision=0, minimum=0, info="Blood Pressure")
    fasting_blood_sugar = gr.Number(label="Fasting Blood Sugar(mg/dL)", precision=0, minimum=0, info="the concentration of glucose (sugar) in the bloodstream after an extended period of fasting")
    cholesterol = gr.Number(label="Total Cholesterol(mg/dL)", precision=0, minimum=0, info="Total amount of cholesterol present in the blood")
    triglyceride = gr.Number(label="Triglyceride(mg/dL)", precision=0, minimum=0, info="A type of fat (lipid) found in blood")
    HDL = gr.Number(label="High-Density Lipoprotein(mg/dL) ", precision=0, minimum=0, info="It is commonly referred to as 'good cholesterol'")
    LDL = gr.Number(label="Low-Density Lipoprotein(mg/dL) ", precision=0, minimum=0, info="It is commonly referred to as 'bad cholesterol'")
    hemoglobin = gr.Number(label="Hemoglobin(g/dL)", minimum=0, info="a protein found in red blood cells that is responsible for carrying oxygen from the lungs to the tissues and organs of the body")
    urine_protein = gr.Radio(label="Does urine contain excessive traces of protein?",choices=[("Yes",2),("No",1)], info="when excessive protein is detected in the urine, it may indicate a problem with kidney function or other underlying health conditions.")
    serum_creatinine = gr.Number(label="Serum creatinine(mg/dL)", minimum=0, info="Serum creatinine levels are commonly measured through a blood test and are used to assess kidney function")
    AST = gr.Number(label="Aspartate Aminotransferase(IU/L)", precision=0, minimum=0, info="glutamic oxaloacetic transaminase type; AST is released into the bloodstream when cells are damaged or destroyed, such as during injury or disease affecting organs rich in AST.")
    ALT = gr.Number(label="Alanine Aminotransferase(IU/L)", precision=0, minimum=0, info="glutamic oxaloacetic transaminase type; ALT is primarily found in the liver cells, and increased levels of ALT in the blood can indicate liver damage or disease")
    Gtp = gr.Number(label="Gamma-glutamyl Transferase(IU/L)", precision=0, minimum=0, info="Elevated levels of GGT in the blood can indicate liver disease or bile duct obstruction. GGT levels are often measured alongside other liver function tests to assess liver health and function.")
    dental_caries = gr.Radio(label="Are there any signs of dental cavities?",choices=[("Yes",1),("No",0)])
    inputs = [age, height, weight, waist, eye_L, eye_R, hear_L, hear_R, systolic, relaxation, fasting_blood_sugar, cholesterol, triglyceride, HDL, LDL, hemoglobin, urine_protein, serum_creatinine, AST, ALT, Gtp, dental_caries]
    smoker_label = gr.Label(label="Predicted Label")

    model_page = gr.Interface(
        predict, 
        inputs=inputs, 
        outputs=smoker_label, 
        examples=load_examples("examples.csv"),
        title="Interact with the Ensemble Classifier Model",
        description="**Medical Disclaimer**: The predictions provided by this model are for educational purposes only and should not be considered a substitute for professional medical advice."
    )

    iface = gr.TabbedInterface(
        [info_page, model_page],
        ["Information", "Smoker Model"]
    )
    
    iface.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/"])

if __name__ == "__main__":
    load_interface()
