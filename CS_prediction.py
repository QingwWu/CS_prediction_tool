import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt



# Load the model
model = joblib.load('xgb.pkl')
# Define feature names
feature_names = [ "Age", "Sex", "Smoking", "Drinking", "hypertension", "Diabetes",'Hyperlipidemia', "BMI","PLT","MPV",
                  "CRP", "PT", "APTT", "TT", 'INR', "FIB", 'D-D']
# Streamlit user interface
st.title("伴右向左分流隐源性卒中风险预测")
# age: numerical input
age = st.number_input("年龄:", min_value=1, max_value=120, value=50)
# sex: categorical selection
sex = st.selectbox("性别 (0=女, 1=男):", options=[0, 1], format_func=lambda x: '女(0)' if x == 0 else '男(1)')
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
        if predicted_proba[predicted_class] > 0.7:
            advice = (f"根据我们的模型，你患隐源性中风的风险很低。" f"该模型预测，你不患隐源性中风的概率是{probability:.1f}%." "然而，保持健康的生活方式仍然非常重要。" 
                   "我建议定期检查以监测你的心脏健康,"  "如果您出现任何症状，请及时就医。")
        else:
            advice = (f"根据我们的模型，你患隐源性中风的风险较低。" f"该模型预测，你不患隐源性中风的概率是{probability:.1f}%." "然而，保持健康的生活方式仍然非常重要。" 
                   "我建议定期检查以监测你的心脏健康,"  "如果您出现任何症状，请及时就医。")
    st.write(advice)
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
