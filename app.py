import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="سامانه هوشمند تشخیص بیماری قلبی", page_icon="🫀", layout="wide")

# ---------- استایل ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; }
    h1, h2, h3, p, li { text-align: right; }
    div.stButton > button {
        width: 100%; height: 3em; font-size: 1.1em; font-weight: bold;
        background: linear-gradient(90deg, #e63946, #d62828);
        color: white; border: none; border-radius: 12px;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 12px; padding: 15px;
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 سامانه هوش مصنوعی توضیح‌پذیر پیش‌بینی بیماری قلبی")
st.markdown("مبتنی بر مطالعه معتبر **Framingham** — با قابلیت توضیح تصمیم مدل برای پزشکان (XAI)")

# ---------- بارگذاری داده و مدل ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/arezu-mashkany/heart-disease-xai/main/heart.csv"
    df = pd.read_csv(url, header=0).dropna()
    return df

df = load_data()

@st.cache_resource
def train_model(data):
    X = data.drop(columns=['TenYearCHD'])
    y = data['TenYearCHD']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_cols = train_model(df)

# ---------- کارت‌های آماری ----------
st.markdown("### 📊 نمای کلی دیتاست")
c1, c2, c3, c4 = st.columns(4)
c1.metric("تعداد بیماران", f"{len(df):,}")
c2.metric("ریسک بالا (TenYearCHD=1)", f"{int(df['TenYearCHD'].sum()):,}")
c3.metric("میانگین سن", f"{df['age'].mean():.0f} سال")
c4.metric("درصد ریسک مثبت", f"{df['TenYearCHD'].mean()*100:.1f} %")

st.markdown("---")

# ---------- تب‌بندی ----------
tab_input, tab_result, tab_guide, tab_about = st.tabs(
    ["📝 ورود اطلاعات بیمار", "🧠 نتیجه و تحلیل XAI", "📖 راهنمای فاکتورها", "ℹ️ درباره پروژه"])

with tab_input:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("سن", int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
        male = st.selectbox("جنسیت", (1, 0), format_func=lambda x: "مرد" if x == 1 else "زن")
        cigsPerDay = st.slider("تعداد سیگار در روز", 0, 70, 0)
        diabetes = st.selectbox("دیابت", (0, 1), format_func=lambda x: "بله" if x == 1 else "خیر")
    with col2:
        totChol = st.slider("کلسترول کل (mg/dL)", 100, 600, int(df['totChol'].mean()))
       sysBP = st.slider("فشار خون سیستولیک", float(df['sysBP'].min()), float(df['sysBP'].max()), float(df['sysBP'].mean()))
diaBP = st.slider("فشار خون دیاستولیک", float(df['diaBP'].min()), float(df['diaBP'].max()), float(df['diaBP'].mean()))
BMI = st.slider("شاخص توده بدنی (BMI)", float(df['BMI'].min()), float(df['BMI'].max()), float(df['BMI'].mean()))

    with col3:
        BMI = st.slider("شاخص توده بدنی (BMI)", 15.0, 60.0, float(df['BMI'].mean()))
        heartRate = st.slider("ضربان قلب", 40, 200, int(df['heartRate'].mean()))
        BPMeds = st.selectbox("مصرف داروی فشار خون", (0, 1), format_func=lambda x: "بله" if x == 1 else "خیر")
        prevalentHyp = st.selectbox("سابقه فشار خون بالا", (0, 1), format_func=lambda x: "بله" if x == 1 else "خیر")

    df_input = pd.DataFrame({
        'male': [male], 'age': [age], 'education': [2.0],
        'currentSmoker': [1 if cigsPerDay > 0 else 0], 'cigsPerDay': [cigsPerDay],
        'BPMeds': [BPMeds], 'prevalentStroke': [0], 'prevalentHyp': [prevalentHyp],
        'diabetes': [diabetes], 'totChol': [totChol], 'sysBP': [sysBP],
        'diaBP': [diaBP], 'BMI': [BMI], 'heartRate': [heartRate], 'glucose': [glucose]
    })

    st.session_state['input'] = df_input
    st.info("✅ اطلاعات ذخیره شد — به تب «نتیجه و تحلیل XAI» برو و دکمه تحلیل را بزن.")

with tab_result:
    if 'input' not in st.session_state:
        st.warning("ابتدا در تب «ورود اطلاعات بیمار» مقادیر را وارد کنید.")
    else:
        if st.button("🔍 تحلیل و پیش‌بینی"):
            df_input = st.session_state['input']
            prob = model.predict_proba(df_input)[0]
            risk = prob[1] * 100

            # گیج ریسک
            st.markdown("### سطح ریسک بیمار")
            if risk >= 50:
                st.error(f"🔴 ریسک بالا: {risk:.1f}%")
            elif risk >= 20:
                st.warning(f"🟠 ریسک متوسط: {risk:.1f}%")
            else:
                st.success(f"🟢 ریسک پایین: {risk:.1f}%")

            st.progress(min(risk / 100, 1.0))

            st.markdown("---")
            st.markdown("### 🧠 عوامل مؤثر بر این پیش‌بینی (SHAP)")
            st.markdown("نمودار زیر نشان می‌دهد هر فاکتور چقدر پیش‌بینی مدل را به سمت «ریسک بالا» یا «ریسک پایین» سوق داده است:")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_input)
                fig = plt.figure(figsize=(9, 4))
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[1], df_input, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values, df_input, plot_type="bar", show=False)
                st.pyplot(fig)
            except Exception:
                st.info("نمودار تحلیلی تولید نشد.")

            # توضیح متنی ساده برای پزشک
            st.markdown("### 💡 خلاصه توضیح برای پزشک")
            tips = []
            if cigsPerDay > 0: tips.append("🚬 سیگار روزانه، ریسک را افزایش می‌دهد.")
            if sysBP > 140: tips.append("🩺 فشار خون سیستولیک بالاست.")
            if BMI > 30: tips.append("⚖️ BMI در محدوده چاقی است.")
            if glucose > 125: tips.append("🍬 قند خون بالاست (احتمال دیابت).")
            if totChol > 240: tips.append("🧪 کلسترول بالاتر از حد نرمال است.")
            if not tips: tips.append("✅ هیچ فاکتور پرخطری در محدوده غیرطبیعی نیست.")
            for t in tips: st.markdown(f"- {t}")

with tab_guide:
    st.markdown("""
    | فاکتور | توضیح پزشکی | محدوده نرمال |
    |---|---|---|
    | سن | افزایش سن، مهم‌ترین عامل غیرقابل‌تغییر ریسک | — |
    | کلسترول کل (totChol) | کلسترول بالا رسوب در عروق را افزایش می‌دهد | کمتر از 200 |
    | فشار سیستولیک (sysBP) | فشار بالای عروق خونی هنگام انقباض قلب | 90 تا 120 |
    | فشار دیاستولیک (diaBP) | فشار هنگام استراحت قلب | 60 تا 80 |
    | BMI | شاخص توده بدنی | 18.5 تا 25 |
    | قند خون (glucose) | قند بالا نشانه دیابت و آسیب عروقی | 70 تا 100 |
    | سیگار روزانه (cigsPerDay) | هرچه بیشتر، ریسک بالاتر | 0 |
    """)

with tab_about:
    st.markdown("""
    **پروژه کارشناسی: تشخیص بیماری قلبی با هوش مصنوعی توضیح‌پذیر (XAI)**
    - 📦 دیتاست: مطالعه Framingham (ساختار قلبی-عروقی معتبر جهانی)
    - 🤖 مدل: RandomForestClassifier
    - 🧠 توضیح‌پذیری: SHAP (SHapley Additive exPlanations)
    - 🚀 استقرار: Streamlit Community Cloud + GitHub
    - ⚠️ توجه: این ابزار صرفاً کمکی/آموزشی است و جایگزین تشخیص پزشک نیست.
    """)

st.caption("پروژه کارشناسی هوش مصنوعی و توضیح‌پذیر (XAI) — ۱۴۰۴")
