# 실행 순서
# 폴더 이동: cd "D:\Codes\my-projects\Dev-Chat_with_Airtable\Chat_with_Airtable_Streamlit_Cloud_v02"
# 가상 환경 생성 (Windows): python -m venv venv
# 패키지 설치: pip install -r requirements.txt
# 스트림릿 실행: streamlit run Chat_with_Airtable_Streamlit_Cloud_v2.py

import os
import json
import re
import traceback
import requests
import pandas as pd
import numpy as np
import openai
import streamlit as st

# ✅ 설정
st.set_page_config(page_title="Chat with Airtable", page_icon="🤖")
RECENT_TURNS_FOR_GPT = 3  # 최근 대화 턴 수 (GPT context에 포함)

# ✅ API 키 로딩
def read_api_key_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        # 파일이 없는 경우 환경 변수에서 시도
        env_var = os.environ.get(file_path.replace(".txt", ""))
        if env_var:
            return env_var
        
        st.error(f"[파일 읽기 오류] {file_path} → {e}")
        return None

# 환경 설정 - 파일이나 환경변수에서 로드
AIRTABLE_API_KEY = read_api_key_from_file("Airtable_Personal_access_token_BIGTURN.txt")
OPENAI_API_KEY = read_api_key_from_file("OpenAI_API_KEY.txt")
openai.api_key = OPENAI_API_KEY

# ✅ Airtable 데이터 로딩 함수
def load_airtable_bases():
    """모든 Airtable 베이스 정보 가져오기"""
    url = "https://api.airtable.com/v0/meta/bases"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    try:
        response = requests.get(url, headers=headers)
        return response.json().get("bases", [])
    except Exception as e:
        st.error(f"❌ 베이스 목록 불러오기 실패: {e}")
        return []

def get_all_tables_in_base(base_id):
    """특정 베이스의 모든 테이블 정보 가져오기"""
    url = f"https://api.airtable.com/v0/meta/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        tables = data.get("tables", [])
        st.write(f"✅ 테이블 {len(tables)}개 불러옴")
        return [(t["name"], t["id"]) for t in tables]
    except Exception as e:
        st.error(f"❌ 테이블 목록 불러오기 실패: {e}")
        return []

@st.cache_data(ttl=3600)
def get_airtable_data(base_id, table_id):
    """특정 베이스의 특정 테이블 데이터 전체 가져오기"""
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    all_records = []
    offset = None

    try:
        with st.spinner(f"테이블 데이터 불러오는 중..."):
            while True:
                params = {"pageSize": 100}
                if offset:
                    params["offset"] = offset

                res = requests.get(url, headers=headers, params=params)
                data = res.json()

                # 에러 응답 시 출력
                if 'error' in data:
                    st.error(f"❌ 에러 발생: {data['error']}")
                    break

                all_records.extend(data.get("records", []))
                offset = data.get("offset")
                if not offset:
                    break

        st.success(f"  ✅ 총 {len(all_records)}개 레코드 불러옴")
        return all_records
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return None

# ✅ 데이터프레임 변환 및 정리 함수
def clean_column_name(col):
    """컬럼명 정제: 특수문자 제거 및 스네이크 케이스로 변환"""
    col = re.sub(r"[^\w\s]", "", col)
    col = col.strip().replace(" ", "_").lower()
    return col

def normalize_table_name(name):
    """테이블명 정제: 특수문자를 언더스코어로 대체"""
    return re.sub(r'\W+', '_', name.strip().lower())

def is_likely_date_column(series):
    """문자열 컬럼 중 날짜 패턴 비율이 높으면 True"""
    if not series.dtype == object:
        return False
    sample = series.dropna().astype(str).head(20)
    match_count = sum(bool(re.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", val)) for val in sample)
    return match_count >= max(3, len(sample) // 2)

def should_exclude_column(col_name):
    """명백히 사람을 의미하는 컬럼들 제외"""
    exclude_keywords = ['_by', 'manager', 'agent', 'consultant', 'email']
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in exclude_keywords)

# ✅ GPT 관련 유틸 함수
def extract_code_blocks(response_text):
    """GPT 응답에서 코드 블록만 추출"""
    match = re.search(r"```(?:python)?\s*([\s\S]+?)```", response_text)
    if not match:
        return response_text  # 코드 블록이 없는 경우 전체 텍스트 반환
    
    code = match.group(1).strip()
    return code

def execute_code(code_str, local_vars):
    """코드 실행 및 결과 반환"""
    if not isinstance(code_str, str):
        return {"success": False, "error": "코드 실행 오류: exec() 인자는 문자열이어야 합니다."}
    try:
        exec(code_str, {}, local_vars)
        result = local_vars.get("result")
        return {"success": True, "result": result}
    except Exception as e:
        error_msg = traceback.format_exc()
        st.error(f"코드 실행 오류: {str(e)}")
        return {"success": False, "error": error_msg}

def ask_gpt(messages):
    """GPT API 호출"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"GPT 요청 실패: {e}")
        return None

def sanitize_colon_spacing(text):
    """마크다운에서 '문장:' 형식을 잘못 인식하는 것을 방지"""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # 링크인 경우 제외 (http://, https:// 등)
        if "://" not in line:
            # 콜론 뒤에 공백이 없는 경우 추가
            line = re.sub(r"(\S):(\S)", r"\1: \2", line)
        cleaned.append(line)
    return "\n".join(cleaned)

# ✅ 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ 대화 렌더링 함수
def render_chat_history():
    st.markdown("---")
    for msg in st.session_state.chat_history:            
        if msg.get("type") == "code":
            sanitized_code = msg["content"]
            # 마크다운 헤더 기호(#) 이스케이프
            sanitized_code = re.sub(r"^(\s*)#{1,6}\s*", r"\1# ", sanitized_code, flags=re.MULTILINE)
            # HTML 태그 이스케이프
            sanitized_code = sanitized_code.replace("<", "&lt;").replace(">", "&gt;")
            # 모든 마크다운 특수문자 이스케이프 처리
            sanitized_code = sanitized_code.replace("*", "\\*")
            sanitized_code = sanitized_code.replace("_", "\\_")
            sanitized_code = sanitized_code.replace("`", "\\`")

            # pre 태그 사용으로 마크다운 형식 완전히 무효화
            st.markdown(f"""
            <pre style='background-color: #e8e8e8; padding: 12px; border-radius: 8px; font-family: monospace; white-space: pre-wrap; margin: 10px 0;'>
{sanitized_code}
            </pre>
            """, unsafe_allow_html=True)

        elif msg["role"] == "user":
            st.markdown(f"""
<div style='text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin: 5px 0; display: inline-block; float: right; clear: both;'>
{msg["content"]}
</div>
""", unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            st.markdown(f"""
<div style='text-align: left; background-color: #F1F0F0; padding: 10px; border-radius: 10px; margin: 5px 0; display: inline-block; float: left; clear: both;'>
{msg["content"]}
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='clear: both;'></div>", unsafe_allow_html=True)

def get_recent_chat_messages(n=RECENT_TURNS_FOR_GPT):
    """최근 n턴의 대화 내용만 반환"""
    recent = st.session_state.chat_history[-n*2:]
    return [
        {"role": m["role"], "content": m["content"]}
        for m in recent if m.get("type") == "text"
    ]

# ===== 메인 UI 시작 =====
st.title("💬 Chat with Airtable")
st.write("에어테이블 데이터에 기반하여 질문하고 응답받을 수 있습니다.")

# ✅ 베이스 선택 UI
bases = load_airtable_bases()
if not bases:
    st.error("사용 가능한 에어테이블 베이스가 없습니다. API 키를 확인하세요.")
    st.stop()

# 베이스가 1개면 자동 선택, 아니면 사용자가 선택
if len(bases) == 1:
    selected_base = bases[0]
    st.success(f"✅ 1개 베이스 자동 선택됨: {selected_base['name']} ({selected_base['id']})")
else:
    base_options = {base["name"]: base for base in bases}
    selected_base_name = st.selectbox("🗂 사용할 베이스를 선택하세요:", list(base_options.keys()))
    selected_base = base_options[selected_base_name]
    st.success(f"✅ 선택된 베이스: {selected_base['name']} ({selected_base['id']})")

# ✅ 선택한 베이스의 테이블 목록 가져오기
base_id = selected_base["id"]
base_name = selected_base["name"]
tables = get_all_tables_in_base(base_id)

if not tables:
    st.warning("선택한 베이스에 테이블이 없습니다.")
    st.stop()

# 테이블 정보 표시
st.write(f"📋 테이블 목록 ({len(tables)}개):")
for i, (table_name, table_id) in enumerate(tables):
    st.write(f"- {table_name} ({table_id})")

# ✅ 모든 테이블의 데이터 가져오기
with st.expander("📦 테이블 데이터 로딩", expanded=False):
    all_data = {}
    progress = st.progress(0)
    
    for i, (table_name, table_id) in enumerate(tables):
        st.write(f"🔄 테이블 로딩 중: {table_name}")
        records = get_airtable_data(base_id, table_id)
        if records:
            all_data[table_name] = records
        
        # 진행률 표시
        progress.progress((i+1)/len(tables))
    
    st.success(f"✅ 모든 테이블 데이터 로딩 완료! 총 {len(all_data)}개 테이블 로드됨")

# ✅ 데이터프레임 변환 및 정리 과정
with st.expander("🔄 데이터프레임 변환", expanded=False):
    dataframes = {}
    
    for table_name, records in all_data.items():
        fields_only = [r.get("fields", {}) for r in records]
        df = pd.DataFrame(fields_only)
        
        if df.empty:
            st.warning(f"⚠️ {table_name}: 데이터가 없거나 필드가 없습니다.")
            continue
            
        # 컬럼 정제
        df.columns = [clean_column_name(c) for c in df.columns]
        
        # NaN-like 딕셔너리 처리
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: np.nan if isinstance(x, dict) and x.get("specialValue") == "NaN" else x
            )
        
        # 변수명으로 사용할 이름 정제
        df_name = normalize_table_name(table_name)
        
        # 변수 등록
        dataframes[df_name] = df
        
        st.write(f"✅ {df_name} ({table_name}): {df.shape[0]} rows, {df.shape[1]} columns")
    
    st.success("🎉 모든 테이블이 DataFrame으로 변환되었습니다.")

# ✅ 날짜 필드 자동 인식 및 변환
with st.expander("📅 날짜 필드 자동 인식", expanded=False):
    datetime_columns_summary = []
    
    for df_name, df in dataframes.items():
        converted_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            name_looks_like_date = any(keyword in col_lower for keyword in 
                                      ['date', 'created', 'expiry', 'changed', 'submitted'])
            value_looks_like_date = is_likely_date_column(df[col])
            
            if (name_looks_like_date or value_looks_like_date) and not should_exclude_column(col):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        converted_cols.append(col)
                        datetime_columns_summary.append((df_name, col))
                except Exception as e:
                    st.write(f"⚠️ [{df_name}] '{col}' 변환 실패: {e}")
        
        if converted_cols:
            st.write(f"✅ {df_name}: 날짜 필드 변환 완료 → {', '.join(converted_cols)}")
    
    # 날짜 파생 필드 자동 생성
    st.write("🧩 날짜 파생 필드 생성 중...")
    
    for df_name, df in dataframes.items():
        added = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_quarter"] = df[col].dt.to_period("Q").astype(str)
                    added.append(col)
                except Exception as e:
                    st.write(f"⚠️ [{df_name}] {col} 파생 필드 생성 실패: {e}")
        
        if added:
            st.write(f"✅ {df_name}: 파생 필드 생성 완료 → {', '.join(added)}")
    
    st.success("🎉 모든 날짜 파생 필드 생성 완료!")

# ✅ 범주형 필드(unique 값 적은 필드) 추출
with st.expander("🧩 범주형 필드 추출", expanded=False):
    categorical_summary = {}
    
    for df_name, df in dataframes.items():
        cat_fields = {}
        
        for col in df.columns:
            if df[col].dtype in ["object", "category"]:
                # 리스트나 딕셔너리 들어있는 컬럼은 제외
                sample_vals = df[col].dropna().head(10)
                if len(sample_vals) > 0 and sample_vals.apply(lambda x: isinstance(x, (list, dict))).any():
                    continue
                
                try:
                    unique_vals = df[col].dropna().unique()
                    if 0 < len(unique_vals) <= 10:
                        cat_fields[col] = list(map(str, unique_vals))
                except Exception as e:
                    st.write(f"⚠️ {df_name} / {col} unique 추출 실패: {e}")
        
        if cat_fields:
            categorical_summary[df_name] = cat_fields
            st.write(f"✅ {df_name}: {len(cat_fields)}개 필드 추출됨")
    
    st.success("✅ 범주형 필드 추출 완료!")

# ✅ 메타데이터 요약 생성
slim_meta_summary = {}

for df_name, df in dataframes.items():
    columns = {}
    for col in df.columns:
        col_data = df[col]
        col_entry = {}
        
        # datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            col_entry["dtype"] = "datetime"
            derived = [f"{col}_year", f"{col}_quarter"]
            col_entry["derived"] = [d for d in derived if d in df.columns]
        
        # categorical
        elif col_data.dtype in ["object", "category"]:
            col_entry["dtype"] = "categorical"
            try:
                sample_vals = col_data.dropna().head(10)
                if len(sample_vals) > 0 and sample_vals.apply(lambda x: isinstance(x, (list, dict))).any():
                    continue
                unique_vals = col_data.dropna().unique()
                if 0 < len(unique_vals) <= 10:
                    col_entry["values"] = list(map(str, unique_vals[:10]))  # 최대 10개 포함
            except:
                continue
        
        # numeric
        elif np.issubdtype(col_data.dtype, np.number):
            col_entry["dtype"] = "numeric"
        
        # 포함 대상만 기록
        if col_entry:
            columns[col] = col_entry
    
    if columns:
        slim_meta_summary[df_name] = {
            "record_count": len(df),
            "columns": columns
        }

# 메타데이터 요약 간략히 표시
with st.expander("📋 메타데이터 요약", expanded=False):
    st.write(json.dumps({k: v for k, v in list(slim_meta_summary.items())[:1]}, indent=2))
    st.success("✅ 메타데이터 생성 완료!")

# ✅ 질문 입력란
user_question = st.chat_input("질문을 입력하세요")

if user_question:
    # 사용자 질문 저장
    st.session_state.chat_history.append({
        "role": "user", "content": user_question, "type": "text"
    })
    
    # GPT 코드 생성 요청
    with st.spinner("🤖 GPT가 질문 분석 중..."):
        # 각 데이터프레임의 실제 모양 정보 추가
        df_shapes = {}
        for df_name, df in dataframes.items():
            # 데이터프레임의 복사본을 생성하여 날짜 객체를 문자열로 변환
            sample_df = df.head(3).copy()
            
            # 날짜 타입 컬럼을 문자열로 변환
            for col in sample_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                    sample_df[col] = sample_df[col].astype(str)
            
            df_shapes[df_name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "sample_data": sample_df.to_dict(orient='records') if not df.empty else {}
            }
            
        prompt = f"""
        다음은 Airtable에서 추출한 테이블 구조 요약입니다:

        {json.dumps(slim_meta_summary, indent=2, ensure_ascii=False)}
        
        데이터프레임 실제 구조 정보:
        {json.dumps(df_shapes, indent=2, ensure_ascii=False)}

        사용자 질문:
        {user_question}

        [분석 환경 안내]
        - 모든 테이블은 이미 pandas DataFrame으로 로딩되어 있습니다.
        - 각 테이블은 snake_case로 정제된 이름의 변수로 존재합니다. 예시: "Client Database" → client_database
        - 데이터를 조회하거나 집계할 때 반드시 이 DataFrame을 기준으로 분석하세요.
        - 실행 결과는 반드시 result 변수에 담고, print() 등은 사용하지 마세요.
        - result 변수는 int, float, str, list, dict 등 간단한 타입으로 설정해주세요.
        - 코드를 작성할 때는 위에 제공된 데이터프레임 실제 구조 정보를 참고하세요.
        - 코드만 반환하고, 설명은 생략하세요.
        """

        messages = [
            {"role": "system", "content": "너는 Airtable 기반 데이터 분석 전문가야."},
            {"role": "user", "content": prompt}
        ]
        
        gpt_response = ask_gpt(messages)
        
        if gpt_response:
            code_str = extract_code_blocks(gpt_response)
            
            # 코드 실행
            local_vars = {name: df for name, df in dataframes.items()}
            result = execute_code(code_str, local_vars)
            
            # 코드 저장
            st.session_state.chat_history.append({
                "type": "code", "content": code_str
            })
            
            if result["success"]:
                # 자연어 해석 요청
                explain_prompt = f"""
                사용자의 질문:
                {user_question}

                코드 실행 결과:
                {result["result"]}

                [지침]
                - 결과를 직접적으로 해석해서 자연스럽게 말해주세요.
                - 질문 내용을 반복하거나 요약하지 말고, 결과 자체에 초점을 맞춰 해석하세요.
                """
                
                explain_messages = [
                    {"role": "system", "content": "친절한 분석가"},
                    {"role": "user", "content": explain_prompt}
                ]
                
                explain_response = ask_gpt(explain_messages)
                
                if explain_response:
                    explain_response = sanitize_colon_spacing(explain_response)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": explain_response, "type": "text"
                    })
                else:
                    st.warning("GPT 설명 생성 실패")
            else:
                error_msg = result.get("error", "알 수 없는 오류")
                st.error("코드 실행 오류 발생")
                
                # 오류 메시지에서 더 친절한 형태로 변환
                simplified_error = re.sub(r'File ".*?", line \d+, in .*?\n', '', error_msg)
                simplified_error = re.sub(r'File "<string>", line \d+, in <module>\n', '', simplified_error)
                
                # 사용자에게 친절한 오류 설명 생성 요청
                error_explain_prompt = f"""
                사용자의 질문: {user_question}
                코드 실행 중 오류가 발생했습니다: {simplified_error}
                
                이 오류의 원인과 해결 방법을 사용자에게 친절하게 설명해주세요.
                """
                error_messages = [{"role": "system", "content": "친절한 데이터 전문가"}]
                error_messages.append({"role": "user", "content": error_explain_prompt})
                error_response = ask_gpt(error_messages)
                
                if error_response:
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": error_response, "type": "text"
                    })
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": f"코드 실행 중 오류가 발생했습니다: {simplified_error}", "type": "text"
                    })
        else:
            st.warning("GPT 응답이 비어 있습니다.")

# UI에 채팅 기록 표시
render_chat_history()

# ✅ 앱 소개 사이드바
with st.sidebar:
    st.title("🤖 Chat with Airtable")
    st.markdown("""
    ## 사용 방법
    1. 에어테이블 베이스가 자동으로 연결됩니다.
    2. 데이터는 pandas DataFrame으로 변환됩니다.
    3. 질문을 입력하면 GPT가 코드를 생성하고 실행합니다.
    4. 결과를 분석하여 답변을 제공합니다.
    
    ## 데이터프레임 정보
    """)
    
    for df_name, df in dataframes.items():
        st.markdown(f"**{df_name}**: {df.shape[0]}행 × {df.shape[1]}열")
