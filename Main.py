import pandas as pd
import pickle 
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Charger le modèle
data = pickle.load(open('Breast_Cancer.sav', 'rb'))

# Titre de l'application
st.title("Breast Cancer Prediction App")

st.info('Use this test to see the model\'s predictions, but remember that this is only a support tool and not a medical diagnosis.')

# Création des champs d'entrée de texte pour chaque caractéristique
radius_mean = float(st.text_input("Average tumor size (in mm)", "0.0"))
texture_mean = float(st.text_input("Average texture of the tumor (smooth or rough)", "0.0"))
perimeter_mean = float(st.text_input("Average perimeter of the tumor (in mm)", "0.0"))
area_mean = float(st.text_input("Average area of the tumor (in mm²)", "0.0"))
smoothness_mean = float(st.text_input("Average smoothness (how smooth the tumor is)", "0.0"))
compactness_mean = float(st.text_input("Average compactness (how tightly packed the tumor is)", "0.0"))
concavity_mean = float(st.text_input("Average concavity (depth of the indentations)", "0.0"))
concave_points_mean = float(st.text_input("Average number of concave points (number of indentations)", "0.0"))
symmetry_mean = float(st.text_input("Average symmetry of the tumor", "0.0"))
fractal_dimension_mean = float(st.text_input("Edge complexity (fractal dimension)", "0.0"))

radius_se = float(st.text_input("Variation in tumor size", "0.0"))
texture_se = float(st.text_input("Variation in texture", "0.0"))
perimeter_se = float(st.text_input("Variation in perimeter", "0.0"))
area_se = float(st.text_input("Variation in area", "0.0"))
smoothness_se = float(st.text_input("Variation in smoothness", "0.0"))
compactness_se = float(st.text_input("Variation in compactness", "0.0"))
concavity_se = float(st.text_input("Variation in concavity", "0.0"))
concave_points_se = float(st.text_input("Variation in concave points", "0.0"))
symmetry_se = float(st.text_input("Variation in symmetry", "0.0"))
fractal_dimension_se = float(st.text_input("Variation in edge complexity (fractal dimension)", "0.0"))

radius_worst = float(st.text_input("Largest tumor size (in mm)", "0.0"))
texture_worst = float(st.text_input("Roughest texture", "0.0"))
perimeter_worst = float(st.text_input("Largest perimeter of the tumor (in mm)", "0.0"))
area_worst = float(st.text_input("Largest area of the tumor (in mm²)", "0.0"))
smoothness_worst = float(st.text_input("Least smooth tumor surface", "0.0"))
compactness_worst = float(st.text_input("Most compact tumor", "0.0"))
concavity_worst = float(st.text_input("Deepest indentations", "0.0"))
concave_points_worst = float(st.text_input("Highest number of concave points", "0.0"))
symmetry_worst = float(st.text_input("Least symmetrical tumor", "0.0"))
fractal_dimension_worst = float(st.text_input("Most complex edge (fractal dimension)", "0.0"))

# Création d'une DataFrame à partir des données d'entrée
df = pd.DataFrame({
        'radius_mean': [radius_mean],
        'texture_mean': [texture_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean],
        'smoothness_mean': [smoothness_mean],
        'compactness_mean': [compactness_mean],
        'concavity_mean': [concavity_mean],
        'concave_points_mean': [concave_points_mean],
        'symmetry_mean': [symmetry_mean],
        'fractal_dimension_mean': [fractal_dimension_mean],
        
        'radius_se': [radius_se],
        'texture_se': [texture_se],
        'perimeter_se': [perimeter_se],
        'area_se': [area_se],
        'smoothness_se': [smoothness_se],
        'compactness_se': [compactness_se],
        'concavity_se': [concavity_se],
        'concave_points_se': [concave_points_se],
        'symmetry_se': [symmetry_se],
        'fractal_dimension_se': [fractal_dimension_se],
        
        'radius_worst': [radius_worst],
        'texture_worst': [texture_worst],
        'perimeter_worst': [perimeter_worst],
        'area_worst': [area_worst],
        'smoothness_worst': [smoothness_worst],
        'compactness_worst': [compactness_worst],
        'concavity_worst': [concavity_worst],
        'concave_points_worst': [concave_points_worst],
        'symmetry_worst': [symmetry_worst],
        'fractal_dimension_worst': [fractal_dimension_worst]
    }, index=[0])

# Standardisation des données avec le même scaler utilisé pendant l'entraînement
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Ajout du bouton de soumission
if st.button('Submit'):
    # Prédiction du modèle
    result = data.predict(df_scaled)
    
    if result == 1:
        st.info("The model predicts a Malignant tumor (cancerous).")
        st.markdown("""
            **Important:** This prediction is based on the model's analysis and should not be considered a definitive diagnosis. It's essential to consult with a medical professional for further evaluation and treatment.

            **General Health Tips:**
            - Maintain a healthy weight.
            - Limit alcohol consumption.
            - Don't smoke.
            - Be physically active.
            - Eat a balanced diet with plenty of fruits and vegetables.
        """)
    else:
        st.info("The model predicts a Benign tumor (non-cancerous).")
        st.markdown("""
            **Important:** While the model suggests a benign tumor, it's crucial to consult with a medical professional for confirmation and appropriate follow-up.

            **General Health Tips:**
            - It's essential to maintain a healthy lifestyle, even if you receive a benign diagnosis.
            - Be aware of any changes in your body and consult with your doctor if you have concerns.
            - Practice regular self-exams and schedule routine checkups.
        """)