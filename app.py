import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# تنظیمات صفحه
st.set_page_config(page_title="تشخیص بیماری قلبی با XAI", layout="wide")

# استایل ظاهری
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; }
    h1, h2, h3, p, li { text-align: right; }
    </style>
    """, unsafe_allow_html=True)

st.title("🫀 سامانه هوش مصنوعی و توضیح‌پذیر (XAI) تشخیص بیماری قلبی")

@st.cache_data
def load_data():
    # آدرس مستقیم فایل خام در گیت‌هاب
    url = "https://raw.githubusercontent.com/arezu-mashkany/heart-disease-xai/main/heart.csv"
    
    # چون فایل شما هدر دارد، header=0 قرار می‌دهیم
    df = pd.read_csv(url, header=0)
    
    # تبدیل ستون‌های متنی به عددی برای اینکه مدل کار کند (Encoding)
    # Sex: M=1, F=0
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    # ChestPainType: تبدیل به عدد
    df['ChestPainType'] = df['ChestPainType'].astype('category').cat.codes
    # RestingECG: تبدیل به عدد
    df['RestingECG'] = df['RestingECG'].astype('category').cat.codes
    # ExerciseAngina: N=0, Y=1
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    # ST_Slope: تبدیل به عدد
    df['ST_Slope'] = df['ST_Slope'].astype('category').cat.codes
    
    # ستون هدف (HeartDisease) را به 0 و 1 استاندارد تبدیل می‌کنیم اگر لازم باشد
    # در اینجا فرض بر این است که HeartDisease یا 0 است یا 1
    
    return df

try:
    df = load_data()
    st.info(f"📊 تعداد نمونه‌های بارگذاری‌شده: {len(df)} بیمار")
except Exception as e:
    st.error(f"خطا در بارگذاری: {e}")
    st.stop()

# بخش ورودی کاربر
st.sidebar.header("مشخصات بالینی بیمار")

def user_input_features():
    # بر اساس ستون‌های فایل شما
    age = st.sidebar.slider("Age (سن)", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))
    sex = st.sidebar.selectbox("Sex (جنسیت: 1:M, 0:F)", (1, 0))
    cp = st.sidebar.selectbox("ChestPainType (کد نوع درد)", sorted(df['ChestPainType'].unique()))
    trestbps = st.sidebar.slider("RestingBP (فشار خون)", int(df['RestingBP'].min()), int(df['RestingBP'].max()), int(df['RestingBP'].mean()))
    chol = st.sidebar.slider("Cholesterol (کلسترول)", int(df['Cholesterol'].min()), int(df['Cholesterol'].max()), int(df['Cholesterol'].mean()))
    fbs = st.sidebar.selectbox("FastingBS (قند خون)", (0, 1))
    recg = st.sidebar.selectbox("RestingECG (کد ECG)", sorted(df['RestingECG'].unique()))
    maxhr = st.sidebar.slider("MaxHR (بیشینه ضربان)", int(df['MaxHR'].min()), int(df['MaxHR'].max()), int(df['MaxHR'].mean()))
    exang = st.sidebar.selectbox("ExerciseAngina (0:No, 1:Yes)", (0, 1))
    oldpeak = st.sidebar.slider("Oldpeak", float(df['Oldpeak'].min()), float(df['Oldpeak'].max()), float(df['Oldpeak'].mean()))
    slope = st.sidebar.selectbox("ST_Slope (کد شیب)", sorted(df['ST_Slope'].unique()))

    data = {
        'Age': age, 'Sex': sex, 'ChestPainType': cp, 'RestingBP': trestbps,
        'Cholesterol': chol, 'FastingBS': fbs, 'RestingECG': recg,
        'MaxHR': maxhr, 'ExerciseAngina': exang, 'Oldpeak': oldpeak,
        'ST_Slope': slope
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

st.subheader("📋 اطلاعات ورودی بیمار:")
st.write(df_input)

# آموزش مدل
@st.cache_resource
def train_model(data):
    # جدا کردن ویژگی‌ها و هدف بر اساس نام ستون‌های جدید شما
    X = data.drop(columns=['HeartDisease'])
    y = data['HeartDisease']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

if st.button("🔍 تحلیل و پیش‌بینی"):
    prediction = model.predict(df_input)
    prob = model.predict_proba(df_input)

    if prediction[0] == 1:
        st.error(f"⚠️ ریسک بالا: احتمال ابتلای به بیماری قلبی وجود دارد. (اطمینان: {prob[0][1]*100:.1f}%)")
    else:
        st.success(f"✅ ریسک پایین: احتمال ابتلای کم است. (اطمینان: {prob[0][0]*100:.1f}%)")

    # XAI Section
    st.markdown("---")
    st.subheader("🧠 تحلیل عوامل مؤثر (XAI)")
    
    explainer = shap.TreeExplainer(model)
    # برای مدل RandomForest در SHAP، استفاده از shap_values[1] برای کلاس مثبت است
    shap_values = explainer.shap_values(df_input)
    
    fig, ax = plt.subplots()
    # اگر shap_values لیست باشد (برای مدل‌های چند کلاسی یا قدیمی)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], df_input, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
    
    st.pyplot(plt.gcf())
