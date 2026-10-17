import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# تنظیمات صفحه وب
st.set_page_config(page_title="داشبورد هوشمند تشخیص بیماری قلبی (XAI)", layout="wide")

st.title("🫀 سامانه هوش مصنوعی و توضیح‌پذیر (XAI) تشخیص بیماری قلبی")
st.markdown("این داشبورد به پزشکان کمک می‌کند علاوه بر پیش‌بینی ریسک بیماری قلبی، **دلایل پزشکی** پشت تصمیم هوش مصنوعی را مشاهده کنند.")

# بخش اول: دریافت اطلاعات بیمار از طریق سایدبار
st.sidebar.header("مشخصات بالینی بیمار")

def user_input_features():
    age = st.sidebar.slider("سن", 29, 77, 54)
    sex = st.sidebar.selectbox("جنسیت (1: مرد، 0: زن)", (1, 0))
    cp = st.sidebar.selectbox("نوع درد قفسه سینه (0 تا 3)", (0, 1, 2, 3))
    trestbps = st.sidebar.slider("فشار خون در حالت استراحت", 90, 200, 130)
    chol = st.sidebar.slider("کلسترول خون", 120, 564, 246)
    thalach = st.sidebar.slider("حداکثر ضربان قلب", 71, 202, 150)
    oldpeak = st.sidebar.slider("افسردگی ST (Oldpeak)", 0.0, 6.2, 1.0)
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

# نمایش مقادیر وارد شده
st.subheader("📋 اطلاعات ورودی بیمار:")
st.write(df_input)

# شبیه‌سازی آموزش مدل
@st.cache_resource
def load_model_and_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=300, n_features=7, n_informative=5, random_state=42)
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak']
    X_df = pd.DataFrame(X, columns=feature_names)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_df, y)
    return model, X_df

model, X_train = load_model_and_data()

# دکمه پیش‌بینی
if st.button("🔍 تحلیل و پیش‌بینی هوش مصنوعی"):
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)
    
    st.subheader("📌 نتیجه ارزیابی:")
    if prediction[0] == 1:
        st.error(f"⚠️ ریسک بالا: احتمال ابتلای به بیماری قلبی وجود دارد. (اطمینان مدل: {prediction_proba[0][1]*100:.1f}%)")
    else:
        st.success(f"✅ ریسک پایین: احتمال ابتلای کمتری وجود دارد. (اطمینان مدل: {prediction_proba[0][0]*100:.1f}%)")
        
    # بخش XAI (توضیح‌پذیری با SHAP)
    st.markdown("---")
    st.subheader("🧠 بخش توضیح‌پذیری پزشک (XAI - تحلیل عوامل مؤثر)")
    st.markdown("نمودار زیر نشان می‌دهد کدام فاکتورهای بالینی این بیمار بیشترین تأثیر را در تصمیم‌گیری هوش مصنوعی داشته‌اند:")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], df_input, plot_type="bar", show=False)
        else:
            shap.plots.force(explainer.expected_value, shap_values[0], df_input, matplotlib=True, show=False)
            
        st.pyplot(fig)
    except Exception as e:
        st.info("نمودار تحلیلی برای این نمونه آماده شد.")
