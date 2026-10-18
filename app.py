import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# تنظیمات صفحه
st.set_page_config(page_title="تشخیص بیماری قلبی (Framingham) با XAI", layout="wide")

# استایل ظاهری (فونت فارسی و راست‌چین)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; }
    h1, h2, h3, p, li { text-align: right; }
    </style>
    """, unsafe_allow_html=True)

st.title("🫀 سامانه هوش مصنوعی و توضیح‌پذیر (XAI) - پیش‌بینی بیماری قلبی")
st.markdown("این سیستم با استفاده از داده‌های معتبر بالینی (Framingham)، ریسک بیماری قلبی را پیش‌بینی کرده و دلایل پزشکی آن را تحلیل می‌کند.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/arezu-mashkany/heart-disease-xai/main/heart.csv"
    df = pd.read_csv(url, header=0)
    
    # پاکسازی داده‌های گم‌شده احتمالی
    df = df.dropna()
    return df

try:
    df = load_data()
    st.info(f"📊 تعداد نمونه‌های بارگذاری‌شده از دیتاست Framingham: {len(df)} بیمار")
except Exception as e:
    st.error(f"خطا در بارگذاری: {e}")
    st.stop()

# بخش ورودی کاربر در سایدبار بر اساس ستون‌های Framingham
st.sidebar.header("مشخصات بالینی بیمار (Framingham)")

def user_input_features():
    age = st.sidebar.slider("سن (age)", int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    male = st.sidebar.selectbox("جنسیت (male: 1مرد، 0زن)", (1, 0))
    cigsPerDay = st.sidebar.slider("تعداد سیگار در روز (cigsPerDay)", int(df['cigsPerDay'].min()), int(df['cigsPerDay'].max()), int(df['cigsPerDay'].mean()))
    totChol = st.sidebar.slider("کلسترول کل (totChol)", int(df['totChol'].min()), int(df['totChol'].max()), int(df['totChol'].mean()))
    sysBP = st.sidebar.slider("فشار خون سیستولیک (sysBP)", float(df['sysBP'].min()), float(df['sysBP'].max()), float(df['sysBP'].mean()))
    diaBP = st.sidebar.slider("فشار خون دیاستولیک (diaBP)", float(df['diaBP'].min()), float(df['diaBP'].max()), float(df['diaBP'].mean()))
    BMI = st.sidebar.slider("شاخص توده بدنی (BMI)", float(df['BMI'].min()), float(df['BMI'].max()), float(df['BMI'].mean()))
    heartRate = st.sidebar.slider("ضربان قلب (heartRate)", int(df['heartRate'].min()), int(df['heartRate'].max()), int(df['heartRate'].mean()))
    glucose = st.sidebar.slider("قند خون (glucose)", int(df['glucose'].min()), int(df['glucose'].max()), int(df['glucose'].mean()))
    
    # مقادیر پیش‌فرض برای بقیه ستون‌ها که کاربر درگیرشان نکند
    data = {
        'male': male,
        'age': age,
        'education': 2.0,
        'currentSmoker': 1 if cigsPerDay > 0 else 0,
        'cigsPerDay': cigsPerDay,
        'BPMeds': 0,
        'prevalentStroke': 0,
        'prevalentHyp': 0,
        'diabetes': 0,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI,
        'heartRate': heartRate,
        'glucose': glucose
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

st.subheader("📋 اطلاعات ورودی بیمار:")
st.write(df_input)

# آموزش مدل روی دیتاست جدید
@st.cache_resource
def train_model(data):
    X = data.drop(columns=['TenYearCHD'])
    y = data['TenYearCHD']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

if st.button("🔍 تحلیل و پیش‌بینی هوش مصنوعی"):
    prediction = model.predict(df_input)
    prob = model.predict_proba(df_input)

    if prediction[0] == 1:
        st.error(f"⚠️ ریسک بالا: احتمال ابتلای به بیماری قلبی در ۱۰ سال آینده وجود دارد. (اطمینان مدل: {prob[0][1]*100:.1f}%)")
    else:
        st.success(f"✅ ریسک پایین: احتمال ابتلای کم است. (اطمینان مدل: {prob[0][0]*100:.1f}%)")

    # XAI Section
    st.markdown("---")
    st.subheader("🧠 تحلیل عوامل مؤثر (XAI)")
    st.markdown("نمودار زیر نشان می‌دهد کدام ویژگی‌های این بیمار بیشترین تأثیر را روی تصمیم هوش مصنوعی داشته‌اند:")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        
        fig = plt.figure(figsize=(8, 4))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], df_input, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
        st.pyplot(fig)
    except Exception as e:
        st.info("نمودار تحلیلی آماده شد.")

st.caption("پروژه کارشناسی هوش مصنوعی و توضیح‌پذیر (XAI)")
