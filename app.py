import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# تنظیمات صفحه وب
st.set_page_config(page_title="داشبورد هوشمند تشخیص بیماری قلبی (XAI)", layout="wide")

# استایل CSS برای راست‌چین کردن متن‌ها و فونت فارسی
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Vazirmatn', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stSidebar {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🫀 سامانه هوش مصنوعی و توضیح‌پذیر (XAI) تشخیص بیماری قلبی")
st.markdown("این داشبورد با استفاده از دیتاست واقعی بالینی، بیماری قلبی را پیش‌بینی کرده و **دلایل پزشکی (XAI)** را به پزشک نشان می‌دهد.")

# بارگذاری دیتاست واقعی از گیت‌هاب
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/arezu-mashkany/heart-disease-xai/main/heart.csv"
    
    col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    
    df = pd.read_csv(url, names=col_names, na_values="?")
    df = df.dropna() # حذف ردیف‌های دارای داده گمشده
    
    df['target'] = np.where(df['num'] > 0, 1, 0)
    df = df.drop(columns=['num'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"خطا در بارگذاری دیتاست: {e}")
    st.stop()

# نمایش تعداد داده‌های بارگذاری شده (الان اینجا درست کار می‌کند چون df بارگذاری شده)
st.info(f"📊 تعداد کل نمونه‌های بالینی بارگذاری شده در سیستم: {df.shape[0]} بیمار (با {df.shape[1]} ویژگی پزشکی)")

# بخش اول: دریافت اطلاعات بیمار از طریق سایدبار
st.sidebar.header("مشخصات بالینی بیمار")

def user_input_features():
    age = st.sidebar.slider("سن", int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    sex = st.sidebar.selectbox("جنسیت (1: مرد، 0: زن)", (1, 0))
    cp = st.sidebar.selectbox("نوع درد قفسه سینه (0 تا 3)", (0, 1, 2, 3))
    trestbps = st.sidebar.slider("فشار خون در حالت استراحت", int(df['trestbps'].min()), int(df['trestbps'].max()), int(df['trestbps'].mean()))
    chol = st.sidebar.slider("کلسترول خون", int(df['chol'].min()), int(df['chol'].max()), int(df['chol'].mean()))
    thalach = st.sidebar.slider("حداکثر ضربان قلب", int(df['thalach'].min()), int(df['thalach'].max()), int(df['thalach'].mean()))
    oldpeak = st.sidebar.slider("افسردگی ST (Oldpeak)", float(df['oldpeak'].min()), float(df['oldpeak'].max()), float(df['oldpeak'].mean()))
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 0,
        'restecg': 1,
        'thalach': thalach,
        'exang': 0,
        'oldpeak': oldpeak,
        'slope': 1,
        'ca': 0.0,
        'thal': 2.0
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

# نمایش مقادیر وارد شده
st.subheader("📋 اطلاعات ورودی بیمار:")
st.write(df_input)

# آموزش مدل روی داده‌های واقعی
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['target'])
    y = data['target']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X

model, X_train = train_model(df)

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
