import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False


# Load the model
model = joblib.load('rf.pkl')
# Define feature options
# cp_options = {1: 'Typical angina (1)', 2: 'Atypical angina (2)', 3: 'Non-anginal pain (3)', 4: 'Asymptomatic (4)'}
# restecg_options = {0: 'Normal (0)', 1: 'ST-T wave abnormality (1)', 2: 'Left ventricular hypertrophy (2)'}
# slope_options = {1: 'Upsloping (1)', 2: 'Flat (2)',  3: 'Downsloping (3)'}
# thal_options = {1: 'Normal (1)',  2: 'Fixed defect (2)', 3: 'Reversible defect (3)'}
# Define feature names
feature_names = [ "Age", "Sex", "Smoking", "Drinking", "hypertension", "Diabetes",'Hyperlipidemia', "BMI","PLT","MPV",
                  "CRP", "PT", "APTT", "TT", 'INR', "FIB", 'D-D']
# Streamlit user interface
st.title("伴右向左分流隐源性卒中风险预测")
# age: numerical input
age = st.number_input("年龄:", min_value=1, max_value=120, value=50)
# sex: categorical selection
sex = st.selectbox("性别 (0=女, 1=男):", options=[0, 1], format_func=lambda x: '女(0)' if x == 0 else '男(1)')
# cp: categorical selection
xiyan = st.selectbox("吸烟史:", options=[0, 1], format_func=lambda x: '不吸烟(0)' if x == 0 else '吸烟(1)')
yinjiu = st.selectbox("饮酒史:", options=[0, 1], format_func=lambda x: '不饮酒(0)' if x == 0 else '饮酒(1)')
gaoxueya = st.selectbox("高血压史:", options=[0, 1], format_func=lambda x: '未患有高血压(0)' if x == 0 else '患有高血压(1)')
tangniao = st.selectbox("糖尿病史:", options=[0, 1], format_func=lambda x: '未患有糖尿病(0)' if x == 0 else '患有糖尿病(1)')
gaozhi = st.selectbox("高脂血症史:", options=[0, 1], format_func=lambda x: '未患有高脂血症(0)' if x == 0 else '患有高脂血症(1)')
# numerical input
BMI = st.number_input("体质量指数:", min_value=10.0, max_value=50.0, value=25.95)
PLT = st.number_input("血小板:", min_value=10, max_value=500, value=191)
MPV = st.number_input("平均血小板体积:", min_value=1.0, max_value=50.0, value=11.9)
CRP = st.number_input("C反应蛋白:", min_value=0.1, max_value=50.0, value=0.5)
PT = st.number_input("凝血酶原时间:", min_value=1.0, max_value=50.0, value=11.5)
APTT = st.number_input("活化部分凝血活酶时间:", min_value=1.0, max_value=50.0, value=26.1)
TT = st.number_input("凝血酶时间:", min_value=1.0, max_value=50.0, value=17.8)
INR = st.number_input("国际标准化比值:", min_value=0.01, max_value=10.0, value=1.05)
FIB = st.number_input("纤维蛋白原:", min_value=1.0, max_value=50.0, value=2.01)
DD = st.number_input("D二聚体:", min_value=10, max_value=2000, value=190)

#
# # chol: numerical input
# chol = st.number_input("Serum cholesterol in mg/dl (chol):", min_value=100, max_value=600, value=200)
# # fbs: categorical selection
# fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')
# # restecg: categorical selection
# restecg = st.selectbox("Resting electrocardiographic results:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
# # thalach: numerical input
# thalach = st.number_input("Maximum heart rate achieved (thalach):", min_value=50, max_value=250, value=150)
# # exang: categorical selection
# exang = st.selectbox("Exercise induced angina (exang):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# # oldpeak: numerical input
# oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)
# # slope: categorical selection
# slope = st.selectbox("Slope of the peak exercise ST segment (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
# # ca: numerical input
# ca = st.number_input("Number of major vessels colored by fluoroscopy (ca):", min_value=0, max_value=4, value=0)
# # thal: categorical selection
# thal = st.selectbox("Thal (thal):", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])
# Process inputs and make predictions
feature_values = [age, sex, xiyan, yinjiu, gaoxueya, tangniao, gaozhi, BMI, PLT, MPV, CRP, PT, APTT, TT,INR, FIB, DD]
features = np.array([feature_values])
if st.button("预测"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results
    st.write(f"**预测类别:** {predicted_class}")
    st.write(f"**预测概率:** {predicted_proba[predicted_class]}")
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        if predicted_proba[predicted_class] >0.8:
            advice = (f"根据我们的模型，你患隐源性中风的风险很高。" f"该模型预测你患隐源性中风的概率为 {probability:.1f}%. "  
                  "虽然这只是一个估计，但它表明你可能面临重大风险。" "我建议你尽快咨询神经内科专家，以进行进一步评估" "确保您得到准确的诊断和必要的治疗。")
        elif predicted_proba[predicted_class] >0.7:
            advice = (
                f"根据我们的模型，你患隐源性中风的风险较高。" f"该模型预测你患隐源性中风的概率为 {probability:.1f}%. "
                "虽然这只是一个估计，但它表明你可能面临重大风险。" "我建议你尽快咨询神经内科专家，以进行进一步评估" "确保您得到准确的诊断和必要的治疗。")
        else :
            advice = (
                f"根据我们的模型，你患隐源性中风的风险略高。" f"该模型预测你患隐源性中风的概率为 {probability:.1f}%. "
                "虽然这只是一个估计，但它表明你可能面临重大风险。" "我建议你尽快咨询神经内科专家，以进行进一步评估" "确保您得到准确的诊断和必要的治疗。")
    else:
        if predicted_proba[predicted_class] < 0.2:
            advice = (f"根据我们的模型，你患隐源性中风的风险很低。" f"该模型预测，你不患隐源性中风的概率是{probability:.1f}%." "然而，保持健康的生活方式仍然非常重要。" 
                   "我建议定期检查以监测你的心脏健康,"  "如果您出现任何症状，请及时就医。")
        else:
            advice = (f"根据我们的模型，你患隐源性中风的风险较低。" f"该模型预测，你不患隐源性中风的概率是{probability:.1f}%." "然而，保持健康的生活方式仍然非常重要。" 
                   "我建议定期检查以监测你的心脏健康,"  "如果您出现任何症状，请及时就医。")
    st.write(advice)
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value[0], shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
