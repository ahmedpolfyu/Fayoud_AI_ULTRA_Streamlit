import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
import io
import os
import time

# إعداد الصفحة
st.set_page_config(
    page_title="Fayoud AI ULTRA - أداة التداول الذكية",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# تعريف نموذج SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# تحميل النموذج المدرب
@st.cache_resource
def load_model():
    try:
        model = SimpleCNN()
        model_path = 'final_model.pt'
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()
                st.success("✅ تم تحميل النموذج المدرب بنجاح")
                return model, True
            except Exception as e:
                st.warning(f"⚠️ خطأ في تحميل النموذج: {e}")
                model.eval()
                return model, False
        else:
            st.warning("⚠️ ملف النموذج غير موجود، استخدام نموذج افتراضي")
            model.eval()
            return model, False
            
    except Exception as e:
        st.error(f"❌ خطأ عام في تحميل النموذج: {e}")
        return None, False

# تحضير الصورة للتحليل
def preprocess_image(image):
    try:
        if image is None:
            return None
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0
        image_tensor = torch.FloatTensor(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        st.error(f"خطأ في معالجة الصورة: {e}")
        return None

def calculate_time_to_next_candle(current_time, timeframe_minutes=1):
    try:
        current_minute_start = current_time.replace(second=0, microsecond=0)
        next_candle_start = current_minute_start + timedelta(minutes=timeframe_minutes)
        time_remaining = (next_candle_start - current_time).total_seconds()
        return max(0, int(time_remaining))
    except:
        return 60

def analyze_chart(image):
    try:
        if image is None:
            return "❌ لم يتم رفع صورة", "", "", ""
        
        model, is_real_model = load_model()
        
        if model is None:
            return "❌ خطأ في تحميل النموذج", "", "", ""
        
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return "❌ خطأ في معالجة الصورة", "", "", ""
        
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        recommendation = "🟢 CALL" if predicted_class == 0 else "🔴 PUT"
        confidence_text = f"{confidence * 100:.1f}%"
        
        current_time = datetime.now()
        time_remaining = calculate_time_to_next_candle(current_time, 1)
        minutes = time_remaining // 60
        seconds = time_remaining % 60
        timer_text = f"{minutes:02d}:{seconds:02d}"
        
        if is_real_model:
            if confidence > 0.8:
                reason = "🎯 نمط فني قوي مكتشف بثقة عالية - إشارة موثوقة للدخول"
            elif confidence > 0.6:
                reason = "⚠️ نمط فني متوسط القوة - يُنصح بالمتابعة الحذرة"
            else:
                reason = "⚡ إشارة ضعيفة - يُنصح بانتظار إشارة أقوى"
        else:
            reason = "⚠️ تحذير: يتم استخدام نموذج تجريبي - النتائج للاختبار فقط"
        
        return recommendation, confidence_text, timer_text, reason
        
    except Exception as e:
        error_msg = f"❌ خطأ في التحليل: {str(e)}"
        return error_msg, "", "", ""

# CSS مخصص للتصميم
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    
    .timer-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #667eea;
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-text {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# العنوان الرئيسي
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3em; margin-bottom: 0.5rem;">🚀 Fayoud AI ULTRA</h1>
    <h2 style="font-size: 1.5em; opacity: 0.9;">أداة التداول الذكية المتقدمة - تحليل دقيق للشارتات</h2>
</div>
""", unsafe_allow_html=True)

# تخطيط الصفحة
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 ارفع صورة الشارت")
    uploaded_file = st.file_uploader(
        "اختر صورة الشارت",
        type=['png', 'jpg', 'jpeg'],
        help="ارفع صورة الشارت المالي للحصول على تحليل فوري"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="صورة الشارت المرفوعة", use_column_width=True)
        
        # زر التحليل
        if st.button("🔍 تحليل الشارت الآن", type="primary", use_container_width=True):
            with st.spinner("جاري تحليل الشارت..."):
                recommendation, confidence, timer, reason = analyze_chart(image)
                
                # حفظ النتائج في session state
                st.session_state.recommendation = recommendation
                st.session_state.confidence = confidence
                st.session_state.timer = timer
                st.session_state.reason = reason

with col2:
    st.markdown("### 📈 نتائج التحليل")
    
    # عرض النتائج بناءً على session state
    if 'recommendation' in st.session_state:
        # عرض التوصية
        st.markdown(f"""
        <div class="result-card">
            <h3>📈 التوصية</h3>
            <h2 style="color: {'#28a745' if 'CALL' in st.session_state.recommendation else '#dc3545'};
                {st.session_state.recommendation}
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # عرض مستوى الثقة
        st.markdown(f"""
        <div class="result-card">
            <h3>🎯 مستوى الثقة</h3>
            <h2 style="color: #667eea;">{st.session_state.confidence}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # عرض العد التنازلي
        st.markdown(f"""
        <div class="timer-display">
            ⏰ الصفقة القادمة خلال: {st.session_state.timer}
        </div>
        """, unsafe_allow_html=True)
        
        # عرض تفسير التحليل
        st.markdown(f"""
        <div class="result-card">
            <h3>📝 تفسير التحليل</h3>
            <p>{st.session_state.reason}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # زر تحديث الوقت (إذا لزم الأمر)
        # تم إزالة زر التحديث التلقائي لتجنب مشاكل التحديث المستمر
        # يمكن إضافة زر تحديث يدوي إذا رغب المستخدم
    
    else:
        st.info("👆 ارفع صورة الشارت أولاً للحصول على التحليل")

# تحذير مهم
st.markdown("""
<div class="warning-text">
    <h4>⚠️ تحذير مهم</h4>
    <p>هذه الأداة مخصصة للأغراض التعليمية والبحثية فقط. لا تستخدمها كأساس وحيد لاتخاذ قرارات الاستثمار. استشر خبير مالي مؤهل قبل اتخاذ أي قرارات استثمارية.</p>
</div>
""", unsafe_allow_html=True)

# معلومات إضافية
st.markdown("""
---
### 🔬 مدعوم بتقنيات الذكاء الاصطناعي المتقدمة
- ⚡ تحليل فوري ودقيق للشارتات المالية
- 🎯 نموذج مدرب على آلاف الشارتات الحقيقية
- ⏰ توقيت دقيق للصفقات القادمة
- 📊 تحليل أنماط الشموع اليابانية

**تم تطوير هذه الأداة بواسطة فريق Fayoud AI**
""")


