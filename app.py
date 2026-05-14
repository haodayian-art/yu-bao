import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io

# ====================== 1. 頁面佈局與 CSS 風格 ======================
st.set_page_config(layout="wide", page_title="通用 AI 數據視覺化儀表板")

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 0rem;}
    div[data-testid="column"] {padding: 0px 8px;}
    .stTextArea textarea {font-size: 14px;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    div[data-testid="stVerticalBlockBorderless"] {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== API 與模型設定 ======================
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("❌ 找不到 GOOGLE_API_KEY！請在 .streamlit/secrets.toml 中設定。")
    st.stop()

def get_available_model():
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        targets = ['models/gemini-1.5-flash', 'models/gemini-1.5-flash-latest', 'models/gemini-pro']
        for t in targets:
            if t in available_models:
                return genai.GenerativeModel(t)
        if available_models:
            return genai.GenerativeModel(available_models[0])
        else:
            raise Exception("無可用模型")
    except Exception as e:
        st.error(f"模型載入失敗：{e}")
        return None

model = get_available_model()
if model is None:
    st.warning("⚠️ 系統無法載入 AI 模型，請檢查 API Key 權限。")
    st.stop()

# ====================== 2. 資料讀取與強化邏輯 ======================
@st.cache_data
def load_any_data(uploaded_file):
    if uploaded_file is None:
        return None
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            try:
                df = pd.read_csv(uploaded_file)
            except:
                df = pd.read_csv(uploaded_file, encoding='big5')
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension == 'xml':
            df = pd.read_xml(uploaded_file)
        else:
            return None

        # 自動強化：嘗試將字串數值轉為數字
        for col in df.columns:
            if any(k in col for k in ['數', '率', '量', '值', '價']) or df[col].dtype == 'object':
                temp_numeric = pd.to_numeric(df[col], errors='coerce')
                if not temp_numeric.isna().all():
                    df[col] = temp_numeric
        
        # 日期自動優化
        date_keywords = ['日期', '時間', '年份', '年度']
        for col in df.columns:
            if any(k in col for k in date_keywords):
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].dt.strftime('%Y-%m-%d')
                except:
                    pass
        return df
    except Exception as e:
        st.error(f"檔案解析失敗：{e}")
        return None

# ====================== 3. 畫面佈局實作 ======================
st.subheader("📊 動態統計：全格式 LLM 驅動數據儀表板")

with st.sidebar:
    st.header("📂 數據源設定")
    uploaded_file = st.file_uploader("上傳檔案 (CSV, Excel, JSON, XML)", type=["csv", "xls", "xlsx", "json", "xml"])
    st.divider()
    if uploaded_file:
        df = load_any_data(uploaded_file)
        if df is not None:
            st.success(f"成功載入: {uploaded_file.name}")
            all_cols = df.columns.tolist()
            st.subheader("圖表欄位選擇")
            x_axis = st.selectbox("選擇 X 軸 (橫軸/圓餅標籤)", all_cols)
            numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
            y_axis = st.selectbox("選擇 Y 軸 (縱軸/圓餅數值)", numeric_cols or all_cols)
            color_axis = st.selectbox("分類依據 (顏色)", [None] + all_cols)
        else:
            df = None
    else:
        df = None

if df is not None:
    main_col_left, main_col_right = st.columns([1, 3])

    with main_col_left:
        with st.container(border=True):
            st.markdown("**查詢輸入 (Query Input)**")
            query_text = st.text_area(label="Query", placeholder="請 AI 分析這份數據...", height=130, label_visibility="collapsed")
            run_analysis = st.button("執行分析 (RUN ANALYSIS)", type="primary", use_container_width=True)

        with st.container(border=True):
            st.markdown("**過濾器 (Filters)**")
            unique_vals = df[x_axis].unique().tolist()
            if len(unique_vals) < 100:
                selected_val = st.multiselect(f"過濾 {x_axis}", unique_vals, default=unique_vals)
                display_df = df[df[x_axis].isin(selected_val)]
            else:
                display_df = df

    with main_col_right:
        top_row_left, top_row_right = st.columns([2, 1])
        
        with top_row_left:
            with st.container(border=True):
                st.markdown(f"**動態數據分析圖：{y_axis} 趨勢**")
                fig_bar = px.bar(display_df, x=x_axis, y=y_axis, color=color_axis, barmode='group', height=300)
                fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        with top_row_right:
            with st.container(border=True):
                st.markdown(f"**數據佔比 (Pie Chart)**")
                # 圓餅圖實作
                fig_pie = px.pie(
                    display_df, 
                    names=x_axis, 
                    values=y_axis, 
                    hole=0.3, # 環形圖設計
                    height=300
                )
                fig_pie.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

        bottom_row_left, bottom_row_right = st.columns([2, 1])

        with bottom_row_left:
            with st.container(border=True):
                st.markdown("**LLM 生成數據解讀報告**")
                if run_analysis:
                    with st.spinner("AI 正在深度分析數據..."):
                        data_context = f"數據摘要:\n{df.describe(include='all').to_string()}\n前5筆資料:\n{df.head().to_string()}"
                        prompt = f"你是一位數據科學家。請分析這份資料：\n{data_context}\n\n使用者問題：{query_text}"
                        try:
                            response = model.generate_content(prompt)
                            st.write(response.text)
                        except Exception as e:
                            st.write(f"分析失敗：{e}")
                else:
                    st.info("請輸入問題並點擊執行分析。")

        with bottom_row_right:
            with st.container(border=True):
                st.markdown("**描述性統計摘要**")
                st.dataframe(display_df[y_axis].describe(), use_container_width=True)

    with st.expander("🔍 檢視原始數據表"):
        st.dataframe(display_df, use_container_width=True)
else:
    st.info("👋 你好 haoda！請從左側上傳數據檔案 (CSV, Excel, JSON 或 XML) 來啟動儀表板。")