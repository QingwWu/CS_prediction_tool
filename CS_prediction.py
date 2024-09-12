import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Load the model
model = joblib.load('rf.pkl')
# Define feature options
feature_names = ["Age", "BMI", 'D-D','INR', "MPV","FIB"]
# Streamlit user interface
st.title("伴右向左分流隐源性卒中风险预测")
# numerical input
age = st.number_input("年龄(Age):", min_value=20, max_value=100, value=36)
BMI = st.number_input("体质量指数(BMI):", min_value=10.0, max_value=50.0, value=24.15)
MPV = st.number_input("平均血小板体积(MPV):", min_value=1.0, max_value=50.0, value=10.30)
INR = st.number_input("国际标准化比值(INR):", min_value=0.01, max_value=10.0, value=0.90)
FIB = st.number_input("纤维蛋白原(FIB):", min_value=1.0, max_value=50.0, value=3.04)
DD = st.number_input("D-二聚体(DD):", min_value=10, max_value=5000, value=380)

# Process inputs and make predictions
feature_values = [age, BMI, DD, INR, MPV, FIB]
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
            advice = (f"根据我们的模型，您患隐源性卒中的风险很高。" f"该模型预测您患隐源性卒中的概率为 {probability:.1f}%。"  
                  "虽然这只是一个估计，但它表明您可能面临重大风险。" "建议您尽快咨询神经内科专家，以进行进一步评估" "确保您得到准确的诊断和必要的治疗。")
        elif predicted_proba[predicted_class] >0.7:
            advice = (
                f"根据我们的模型，您患隐源性卒中的风险较高。" f"该模型预测您患隐源性卒中的概率为 {probability:.1f}%。"
                "虽然这只是一个估计，但它表明您可能面临较大风险。" "建议您尽快咨询神经内科专家，以进行进一步评估" "确保您得到准确的诊断和必要的治疗。")
        else :
            advice = (
                f"根据我们的模型，您患隐源性卒中的风险略高。" f"该模型预测您患隐源性卒中的概率为 {probability:.1f}%。 "
                "虽然这只是一个估计，但它表明您可能面临风险。" "建议您咨询神经内科专家，以进行进一步评估" "确保您得到准确的诊断和必要的治疗。")
    else:
        if predicted_proba[predicted_class] > 0.8:
            advice = (f"根据我们的模型，您患隐源性卒中的风险很低。" f"该模型预测您患隐源性卒中的概率是{100-probability:.1f}%。" "然而，保持健康的生活方式仍然非常重要。" 
                   "建议您定期检查以监测您的心脏健康，"  "如果您出现任何症状，请及时就医。")
        else:
            advice = (f"根据我们的模型，您患隐源性卒中的风险较低。" f"该模型预测您患隐源性卒中的概率是{100-probability:.1f}%。" "然而，保持健康的生活方式仍然非常重要。" 
                   "建议您定期检查以监测您的心脏健康，"  "如果您出现任何症状，请及时就医。")
    st.write(advice)
    # Calculate SHAP values and display force plot
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    # shap.force_plot(explainer.expected_value[0], shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    # st.image("shap_force_plot.png")
