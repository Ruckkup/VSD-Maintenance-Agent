# app.py
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os # <--- ADDED THIS IMPORT

# --- ตั้งค่าเริ่มต้น ---
st.set_page_config(page_title="VSD Virtual Mentor", layout="wide")
st.title("👨‍🔧 VSD Maintenance: Virtual Mentor")
st.markdown("ผู้ช่วยวิศวกรเสมือนสำหรับตอบปัญหาการบำรุงรักษา Variable Speed Drive")

# ตั้งค่า API Key ของ Google
# แนะนำให้ใช้ st.secrets สำหรับการใช้งานจริง
# For testing, you can paste your key directly
# GOOGLE_API_KEY = "AIzaSyDM0N01ki1RbULaMPK0aObuTNLl9weVyqU"
GOOGLE_API_KEY = "" # Initialize to avoid NameError
try:
    from google_api_key import GOOGLE_API_KEY
except ImportError:
    GOOGLE_API_KEY = st.text_input("กรุณาใส่ Google API Key ของคุณ:", type="password")

if not GOOGLE_API_KEY:
    st.warning("กรุณาใส่ Google API Key เพื่อเริ่มต้นใช้งาน")
    st.stop()
    
# --- ฟังก์ชันหลักของ RAG ---
@st.cache_resource # Cache resource เพื่อไม่ต้องโหลดโมเดลใหม่ทุกครั้ง
def load_rag_pipeline():
    # 1. โหลด Embedding model (ต้องเป็นตัวเดียวกับตอน ingest)
    model_name = "intfloat/multilingual-e5-large"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 2. โหลด Vector Database ที่สร้างไว้
    vectordb = Chroma(persist_directory="./vs_db", embedding_function=embeddings)

    # 3. ตั้งค่า LLM (Gemini Pro)
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

    # 4. สร้าง Prompt Template เพื่อ "สอน" AI ให้เป็นผู้ช่วยวิศวกร
    prompt_template = """
    คุณคือผู้เชี่ยวชาญด้านการบำรุงรักษา VSD (Variable Speed Drive) ระดับซีเนียร์
    หน้าที่ของคุณคือให้คำแนะนำที่ถูกต้อง แม่นยำ และปลอดภัยแก่วิศวกรหรือช่างเทคนิค
    ใช้ข้อมูลจากบริบทที่ให้มาด้านล่างนี้เพื่อตอบคำถามเท่านั้น อย่าสร้างคำตอบขึ้นมาเอง

    **สำคัญ:**
    - หากมีขั้นตอนที่เกี่ยวข้องกับความปลอดภัย (เช่น การตัดไฟ, Lockout-Tagout) ให้เน้นย้ำเสมอ
    - ตอบเป็นขั้นตอนที่ชัดเจน (step-by-step) ถ้าเป็นไปได้
    - ถ้าข้อมูลในบริบทไม่เพียงพอที่จะตอบคำถาม ให้ตอบว่า "ผมไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล โปรดตรวจสอบจากคู่มือโดยตรง"

    **บริบท:**
    {context}

    **คำถาม:**
    {question}

    **คำตอบที่เป็นประโยชน์:**
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 5. สร้าง RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={'k': 3}), # ดึงข้อมูลที่เกี่ยวข้องมา 3 ชิ้น
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- ส่วนของหน้าจอแอปพลิเคชัน ---
rag_pipeline = load_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงข้อความแชทเก่า
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# รับ Input จากผู้ใช้
if prompt := st.chat_input("ถามปัญหาเกี่ยวกับ VSD ได้เลย..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    full_response = "" # Initialize full_response before the try block
    with st.chat_message("assistant"):
        # --- DEBUGGING STARTS HERE ---
        st.write("1. ได้รับคำถามแล้ว กำลังจะเรียก RAG pipeline...")
        
        try:
            with st.spinner("กำลังค้นหาข้อมูลและเรียบเรียงคำตอบ..."):
                st.write("2. เข้าสู่ RAG pipeline... กำลังเรียก retriever เพื่อค้นหาข้อมูล...")
                
                # We can remove the intermediate debugging steps if the core functionality works.
                # These were useful for initial debugging but might clutter the UI in production.
                # st.write("3. ค้นหาข้อมูลเจอแล้ว ... กำลังจะส่งไปให้ LLM...")

                # รัน pipeline ทั้งหมด
                response = rag_pipeline({"query": prompt})
                
                st.write("4. ได้รับคำตอบจาก LLM เรียบร้อยแล้ว! กำลังจะแสดงผล...")

                full_response = response['result']
                
                # แสดงแหล่งอ้างอิง
                source_docs = response['source_documents']
                if source_docs:
                    full_response += "\n\n---\n**แหล่งข้อมูลอ้างอิง:**\n"
                    unique_sources = set([doc.metadata['source'] for doc in source_docs if 'source' in doc.metadata]) # Added check for 'source' key
                    for source in unique_sources:
                        full_response += f"- {os.path.basename(source)}\n"
                
                st.markdown(full_response)
        
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดร้ายแรงขึ้น: {e}")
            full_response = "ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำถามของคุณ โปรดลองอีกครั้งในภายหลัง" # Provide a fallback message

        st.write("5. จบการทำงานในส่วน assistant")
        # --- DEBUGGING ENDS HERE ---
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})