# Execution Order
# Change directory: cd "D:\Codes\my-projects\Dev-Chat_with_Airtable\Chat_with_Airtable_Streamlit_Cloud_v03"
# Create virtual environment (Windows): python -m venv venv
# Install packages: pip install -r requirements.txt
# Run Streamlit: streamlit run Chat_with_Airtable_Streamlit_Cloud_v2.py

import os
import json
import re
import traceback
import requests
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import math
import collections
import openai

# ✅ Configuration
st.set_page_config(page_title="Chat with Airtable", page_icon="🤖")
RECENT_TURNS_FOR_GPT = 3  # Recent conversation turns (included in GPT context)

# ✅ API Key Loading
def read_api_key_from_file(file_path):
    # Extract key name (remove .txt from filename)
    key_name = file_path.replace(".txt", "")
    
    try:
        # 1. Find key in Streamlit secrets - check in global section
        if hasattr(st, 'secrets'):
            if 'global' in st.secrets and key_name in st.secrets['global']:
                print(f"✅ Loaded {key_name} key from global section in Streamlit secrets")
                return st.secrets['global'][key_name]
            elif key_name in st.secrets:
                print(f"✅ Loaded {key_name} key from Streamlit secrets")
                return st.secrets[key_name]
        
        # 2. Find key in file (for local development)
        with open(file_path, 'r') as file:
            key = file.read().strip()
            print(f"✅ Loaded key from {file_path} file")
            return key
            
    except FileNotFoundError:
        # 3. Find key in environment variables
        env_var = os.environ.get(key_name)
        if env_var:
            print(f"✅ Loaded key from environment variable {key_name}")
            return env_var
        
        # All attempts failed
        st.error(f"⚠️ Cannot find {file_path} file, and environment variable is not set")
        return None
        
    except Exception as e:
        st.error(f"⚠️ Key loading error: {file_path} → {e}")
        return None

# Secret access test code
st.write("### Streamlit Secrets Test")
try:
    if hasattr(st, 'secrets'):
        st.write("✅ st.secrets exists")
        st.write(f"Keys in st.secrets: {list(st.secrets.keys())}")
        
        if 'global' in st.secrets:
            st.write("✅ global section exists")
            st.write(f"Keys in global section: {list(st.secrets['global'].keys())}")
            
            if 'Airtable_Personal_access_token_BIGTURN' in st.secrets['global']:
                st.write("✅ Airtable key exists in global section")
                st.write(f"First 5 characters of Airtable key: {st.secrets['global']['Airtable_Personal_access_token_BIGTURN'][:5]}...")
            else:
                st.write("❌ Airtable key does not exist in global section")
                
            if 'OpenAI_API_KEY' in st.secrets['global']:
                st.write("✅ OpenAI key exists in global section")
                st.write(f"First 5 characters of OpenAI key: {st.secrets['global']['OpenAI_API_KEY'][:5]}...")
            else:
                st.write("❌ OpenAI key does not exist in global section")
        else:
            st.write("❌ global section does not exist")
    else:
        st.write("❌ st.secrets does not exist")
except Exception as e:
    st.write(f"❌ Error occurred: {str(e)}")

# Environment setup - load from file or environment variables
AIRTABLE_API_KEY = read_api_key_from_file("Airtable_Personal_access_token_BIGTURN.txt")
OPENAI_API_KEY = read_api_key_from_file("OpenAI_API_KEY.txt")
openai.api_key = OPENAI_API_KEY

# ✅ Airtable Data Loading Functions
def load_airtable_bases():
    """Get all Airtable base information"""
    url = "https://api.airtable.com/v0/meta/bases"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    try:
        response = requests.get(url, headers=headers)
        return response.json().get("bases", [])
    except Exception as e:
        st.error(f"❌ Failed to load base list: {e}")
        return []

def get_all_tables_in_base(base_id):
    """Get all table information for a specific base"""
    url = f"https://api.airtable.com/v0/meta/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        tables = data.get("tables", [])
        st.write(f"✅ Loaded {len(tables)} tables")
        return [(t["name"], t["id"]) for t in tables]
    except Exception as e:
        st.error(f"❌ Failed to load table list: {e}")
        return []

@st.cache_data(ttl=3600)
def get_airtable_data(base_id, table_id):
    """Get all data for a specific table in a base"""
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    all_records = []
    offset = None

    try:
        with st.spinner(f"Loading table data..."):
            while True:
                params = {"pageSize": 100}
                if offset:
                    params["offset"] = offset

                res = requests.get(url, headers=headers, params=params)
                data = res.json()

                # Output if error response
                if 'error' in data:
                    st.error(f"❌ Error occurred: {data['error']}")
                    break

                all_records.extend(data.get("records", []))
                offset = data.get("offset")
                if not offset:
                    break

        st.success(f"  ✅ Loaded {len(all_records)} records total")
        return all_records
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return None

# ✅ DataFrame Conversion and Cleanup Functions
def clean_column_name(col):
    """Clean column name: remove special characters and convert to snake case"""
    col = re.sub(r"[^\w\s]", "", col)
    col = col.strip().replace(" ", "_").lower()
    return col

def normalize_table_name(name):
    """Clean table name: replace special characters with underscores"""
    return re.sub(r'\W+', '_', name.strip().lower())

def is_likely_date_column(series):
    """Return True if string column has high ratio of date patterns"""
    if not series.dtype == object:
        return False
    sample = series.dropna().astype(str).head(20)
    match_count = sum(bool(re.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", val)) for val in sample)
    return match_count >= max(3, len(sample) // 2)

def should_exclude_column(col_name):
    """Exclude columns that clearly refer to people"""
    exclude_keywords = ['_by', 'manager', 'agent', 'consultant', 'email']
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in exclude_keywords)

# ✅ GPT Related Utility Functions
def extract_code_blocks(response_text):
    """Extract only code blocks from GPT response"""
    match = re.search(r"```(?:python)?\s*([\s\S]+?)```", response_text)
    if not match:
        return response_text  # Return entire text if no code blocks found
    
    code = match.group(1).strip()
    return code

def execute_code(code_str, local_vars):
    """Execute code and return results"""
    if not isinstance(code_str, str):
        return {"success": False, "error": "Code execution error: exec() argument must be a string."}
    try:
        exec(code_str, {}, local_vars)
        result = local_vars.get("result")
        return {"success": True, "result": result}
    except Exception as e:
        error_msg = traceback.format_exc()
        st.error(f"Code execution error: {str(e)}")
        return {"success": False, "error": error_msg}

def ask_gpt(messages):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Modify to match actual model name
            messages=messages,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"GPT request failed: {e}")
        return None

def sanitize_colon_spacing(text):
    """Prevent misinterpretation of 'sentence:' format in markdown"""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # Exclude links (http://, https://, etc)
        if "://" not in line:
            # Add space after colon if missing
            line = re.sub(r"(\S):(\S)", r"\1: \2", line)
        cleaned.append(line)
    return "\n".join(cleaned)

# ✅ Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Chat Rendering Function
def render_chat_history():
    st.markdown("---")
    for msg in st.session_state.chat_history:            
        if msg.get("type") == "code":
            sanitized_code = msg["content"]
            # Escape markdown header symbols (#)
            sanitized_code = re.sub(r"^(\s*)#{1,6}\s*", r"\1# ", sanitized_code, flags=re.MULTILINE)
            # Escape HTML tags
            sanitized_code = sanitized_code.replace("<", "&lt;").replace(">", "&gt;")
            # Escape all markdown special characters
            sanitized_code = sanitized_code.replace("*", "\\*")
            sanitized_code = sanitized_code.replace("_", "\\_")
            sanitized_code = sanitized_code.replace("`", "\\`")

            # Use pre tags to completely disable markdown formatting
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
    """Return only the last n turns of conversation"""
    recent = st.session_state.chat_history[-n*2:]
    return [
        {"role": m["role"], "content": m["content"]}
        for m in recent if m.get("type") == "text"
    ]

# ===== Main UI Start =====
st.title("💬 Chat with Airtable")
st.write("You can ask questions and get responses based on your Airtable data.")

# ✅ Base Selection UI
bases = load_airtable_bases()
if not bases:
    st.error("No Airtable bases available. Please check your API key.")
    st.stop()

# Auto-select if only one base, otherwise user selects
if len(bases) == 1:
    selected_base = bases[0]
    st.success(f"✅ Auto-selected 1 base: {selected_base['name']} ({selected_base['id']})")
else:
    base_options = {base["name"]: base for base in bases}
    selected_base_name = st.selectbox("🗂 Select a base to use:", list(base_options.keys()))
    selected_base = base_options[selected_base_name]
    st.success(f"✅ Selected base: {selected_base['name']} ({selected_base['id']})")

# ✅ Get table list for selected base
base_id = selected_base["id"]
base_name = selected_base["name"]
tables = get_all_tables_in_base(base_id)

if not tables:
    st.warning("No tables in selected base.")
    st.stop()

# Display table information
st.write(f"📋 Table list ({len(tables)} tables):")
for i, (table_name, table_id) in enumerate(tables):
    st.write(f"- {table_name} ({table_id})")

# ✅ Load data for all tables
with st.expander("📦 Table Data Loading", expanded=False):
    all_data = {}
    progress = st.progress(0)
    
    for i, (table_name, table_id) in enumerate(tables):
        st.write(f"🔄 Loading table: {table_name}")
        records = get_airtable_data(base_id, table_id)
        if records:
            all_data[table_name] = records
        
        # Show progress
        progress.progress((i+1)/len(tables))
    
    st.success(f"✅ All table data loading complete! Loaded {len(all_data)} tables total")

# ✅ DataFrame Conversion and Cleanup Process
with st.expander("🔄 DataFrame Conversion", expanded=False):
    dataframes = {}
    
    for table_name, records in all_data.items():
        fields_only = [r.get("fields", {}) for r in records]
        df = pd.DataFrame(fields_only)
        
        if df.empty:
            st.warning(f"⚠️ {table_name}: No data or fields present.")
            continue
            
        # Clean columns
        df.columns = [clean_column_name(c) for c in df.columns]
        
        # Handle NaN-like dictionaries
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: np.nan if isinstance(x, dict) and x.get("specialValue") == "NaN" else x
            )
        
        # Clean name for variable use
        df_name = normalize_table_name(table_name)
        
        # Register variable
        dataframes[df_name] = df
        
        st.write(f"✅ {df_name} ({table_name}): {df.shape[0]} rows, {df.shape[1]} columns")
    
    st.success("🎉 All tables have been converted to DataFrames.")

# ✅ Auto-detect and convert date fields
with st.expander("📅 Date Field Auto-Detection", expanded=False):
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
                    st.write(f"⚠️ [{df_name}] '{col}' conversion failed: {e}")
        
        if converted_cols:
            st.write(f"✅ {df_name}: Date field conversion complete → {', '.join(converted_cols)}")
    
    # Auto-generate date derivative fields
    st.write("🧩 Generating date derivative fields...")
    
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
                    st.write(f"⚠️ [{df_name}] Failed to create derivative fields for {col}: {e}")
        
        if added:
            st.write(f"✅ {df_name}: Derivative field creation complete → {', '.join(added)}")
    
    st.success("🎉 All date derivative fields creation complete!")

# ✅ Extract categorical fields (fields with few unique values)
with st.expander("🧩 Categorical Field Extraction", expanded=False):
    categorical_summary = {}
    
    for df_name, df in dataframes.items():
        cat_fields = {}
        
        for col in df.columns:
            if df[col].dtype in ["object", "category"]:
                # Exclude columns containing lists or dictionaries
                sample_vals = df[col].dropna().head(10)
                if len(sample_vals) > 0 and sample_vals.apply(lambda x: isinstance(x, (list, dict))).any():
                    continue
                
                try:
                    unique_vals = df[col].dropna().unique()
                    if 0 < len(unique_vals) <= 10:
                        cat_fields[col] = list(map(str, unique_vals))
                except Exception as e:
                    st.write(f"⚠️ {df_name} / {col} unique extraction failed: {e}")
        
        if cat_fields:
            categorical_summary[df_name] = cat_fields
            st.write(f"✅ {df_name}: {len(cat_fields)} fields extracted")
    
    st.success("✅ Categorical field extraction complete!")

# ✅ Generate Metadata Summary
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
                    col_entry["values"] = list(map(str, unique_vals[:10]))  # Include max 10
            except:
                continue
        
        # numeric
        elif np.issubdtype(col_data.dtype, np.number):
            col_entry["dtype"] = "numeric"
        
        # Record only if entry exists
        if col_entry:
            columns[col] = col_entry
    
    if columns:
        slim_meta_summary[df_name] = {
            "record_count": len(df),
            "columns": columns
        }

# Display metadata summary briefly
with st.expander("📋 Metadata Summary", expanded=False):
    st.write(json.dumps({k: v for k, v in list(slim_meta_summary.items())[:1]}, indent=2))
    st.success("✅ Metadata generation complete!")

# ✅ Question Input
user_question = st.chat_input("Enter your question")

# GPT Code Generation and Execution Logic
# If user question exists
if user_question:
    # Save user question
    st.session_state.chat_history.append({
        "role": "user", "content": user_question, "type": "text"
    })
    
    # Request GPT code generation
    with st.spinner("🤖 GPT is analyzing the question..."):
        # Add actual shape information for each dataframe
        df_shapes = {}
        for df_name, df in dataframes.items():
            sample_df = df.head(3).copy()
            
            # Convert date type columns to strings
            for col in sample_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                    sample_df[col] = sample_df[col].astype(str)
            
            df_shapes[df_name] = {
                "rows": len(df),
                "columns": list(df.columns),
                "sample_data": sample_df.to_dict(orient='records') if not df.empty else {}
            }
            
        prompt = f"""
        Here is a summary of the table structures extracted from Airtable:

        {json.dumps(slim_meta_summary, indent=2, ensure_ascii=False)}
        
        Actual DataFrame structure information:
        {json.dumps(df_shapes, indent=2, ensure_ascii=False)}

        User question:
        {user_question}

        [Analysis Environment Information]
        - All tables are already loaded as pandas DataFrames.
        - The following libraries are already imported:
          - pandas as pd
          - numpy as np
          - datetime
          - re
          - math
          - collections
        - Each table exists as a variable with a snake_case cleaned name. Example: "Client Database" → client_database
        - Always analyze based on these DataFrames when querying or aggregating data.
        - Always store execution results in the 'result' variable, and don't use print() etc.
        - The 'result' variable should be set to a simple type like int, float, str, list, dict, etc.
        - When writing code, refer to the actual DataFrame structure information provided above.
        - Only return the code, without any explanations.
        """
        
        messages = [
            {"role": "system", "content": "You are an Airtable-based data analysis expert."},
            {"role": "user", "content": prompt}
        ]
        
        # Set maximum retry count
        max_retries = 3
        retry_count = 0
        code_success = False
        
        while retry_count < max_retries and not code_success:
            if retry_count > 0:
                st.info(f"Retrying due to code execution error... ({retry_count}/{max_retries})")
            
            # Request code from GPT
            gpt_response = ask_gpt(messages)
            
            if not gpt_response:
                st.warning("GPT response is empty.")
                break
                
            code_str = extract_code_blocks(gpt_response)
            
            if not code_str:
                st.warning("Cannot find code block.")
                break
            
            # Save code (update if previous code exists)
            if retry_count == 0:
                st.session_state.chat_history.append({
                    "type": "code", "content": code_str
                })
            else:
                # Update previous code
                st.session_state.chat_history[-1]["content"] = code_str
            
            # Execute code
            local_vars = {
                'pd': pd, 
                'np': np, 
                'datetime': datetime, 
                're': re,
                'math': math, 
                'collections': collections,
                **{name: df for name, df in dataframes.items()}
            }
            
            try:
                exec(code_str, {}, local_vars)
                result = local_vars.get("result")
                code_success = True
            except Exception as e:
                error_msg = traceback.format_exc()
                
                # If error occurred and retries remain
                if retry_count < max_retries - 1:
                    # Create simplified error message
                    simplified_error = re.sub(r'File ".*?", line \d+, in .*?\n', '', error_msg)
                    simplified_error = re.sub(r'File "<string>", line \d+, in <module>\n', '', simplified_error)
                    
                    # Request error fix from GPT
                    fix_prompt = f"""
                    User's question: {user_question}
                    
                    Your original code:
                    ```python
                    {code_str}
                    ```
                    
                    The following error occurred during execution:
                    ```
                    {simplified_error}
                    ```
                    
                    Please provide corrected code. Pay attention to the following:
                    - Include the entire code in a code block (```)
                    - Identify the cause of the error and fix it properly
                    - pandas is already imported as 'pd', numpy as 'np'
                    - The result must be stored in the 'result' variable
                    - Only return the code, without any explanations
                    """
                    
                    messages = [
                        {"role": "system", "content": "You are an Airtable-based data analysis expert."},
                        {"role": "user", "content": fix_prompt}
                    ]
                else:
                    # Maximum retry count reached
                    st.error(f"Maximum retry count ({max_retries}) reached. Failed to execute code.")
            
            retry_count += 1
        
        # Request natural language explanation after successful code execution
        if code_success:
            # Request natural language interpretation
            explain_prompt = f"""
            User's question:
            {user_question}

            Code execution result:
            {result}

            [Instructions]
            - Interpret the result directly and naturally.
            - Don't repeat or summarize the question, focus on interpreting the result itself.
            """
            
            explain_messages = [
                {"role": "system", "content": "Friendly analyst"},
                {"role": "user", "content": explain_prompt}
            ]
            
            explain_response = ask_gpt(explain_messages)
            
            if explain_response:
                explain_response = sanitize_colon_spacing(explain_response)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": explain_response, "type": "text"
                })
            else:
                st.warning("GPT explanation generation failed")
        else:
            # Failed after all retries
            error_explain_prompt = f"""
            User's question: {user_question}
            
            Code execution has consistently failed. Please kindly explain the cause of the problem and 
            how the user could modify their question or approach the data differently.
            """
            
            error_messages = [
                {"role": "system", "content": "Friendly data expert"},
                {"role": "user", "content": error_explain_prompt}
            ]
            
            error_response = ask_gpt(error_messages)
            
            if error_response:
                st.session_state.chat_history.append({
                    "role": "assistant", "content": error_response, "type": "text"
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant", "content": "Code execution failed. Please make your question more specific or try a different approach.", 
                    "type": "text"
                })

# Display chat history in UI
render_chat_history()

# ✅ App Introduction Sidebar
with st.sidebar:
    st.title("🤖 Chat with Airtable")
    st.markdown("""
    ## How to Use
    1. Airtable bases are automatically connected.
    2. Data is converted to pandas DataFrames.
    3. When you enter a question, GPT generates and executes code.
    4. Results are analyzed to provide an answer.
    
    ## DataFrame Information
    """)
    
    for df_name, df in dataframes.items():
        st.markdown(f"**{df_name}**: {df.shape[0]} rows × {df.shape[1]} columns")