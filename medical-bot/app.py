import os
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from helper import download_hugging_face_embeddings, load_gemini_llm

# Initialize Flask with explicit static and template settings
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app) 

load_dotenv()

# Ensure Pinecone API Key is set in the environment for PineconeVectorStore
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
else:
    print("Warning: PINECONE_API_KEY is not set. Pinecone vector store may fail to initialize.")

INDEX_NAME = "medicalbot"
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 4})

# 🟢 تم تحديث الـ API Key الجديد هنا بنجاح لفك حظر الـ Quota فوراً
llm = load_gemini_llm(api_key="AQ.Ab8RN6KRXt830ZIerDMDSUGg0YUh-BOqbFqBMvUBFnBTkBkRoQ")

# المرجع الطبي العربي المحقون برمجياً مباشرة (بدون ملفات)
ARABIC_MEDICAL_REFERENCE = """
=== مرجع منظمة الصحة العالمية والدليل الطبي العربي الموحد ===
1. مؤشرات خطورة ضغط الدم والسكري (Clinical Risk Vitals):
- المعدل الطبيعي لضغط الدم الانقباضي هو أقل من 120 ملم زئبقي. فوق 140 ملم زئبقي مع صداع يصنف كخطورة متوسطة إلى عالية (Mid to High Risk).
- المعدل الطبيعي للسكر الصائم هو أقل من 100 ملجم/دسل. فوق 200 ملجم/دسل عشوائي مع إعياء يصنف كخطورة عالية (High Risk).
2. بروتوكول التعامل مع الطوارئ والأعراض الحادة:
- آلام الصدر وضيق التنفس الحاد: تصنف فوراً كـ "حالة حرجة خطيرة جداً" (High Risk) والبروتوكول الإلزامي هو التوجيه لأقرب مستشفى فوراً وعدم الانتظار.
"""

# الـ Prompt المتكامل مع فرض قاعدة اللغة الصارمة في البداية لضمان عدم الخلط
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "LANGUAGE RULE: You must always reply in the EXACT same language used by the user in their input. If the user asks in English, reply in English. If they ask in Arabic, reply in Arabic. Do not switch languages based on the context text.\n\n"
        "You are an expert clinical medical assistant operating within a Hybrid AI Clinical System (Pulse).\n"
        "You balance quantitative data from a Predictive ML Model and semantic context from verified reference documents.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Read the 'Patient ML Risk Assessment' context carefully. If it denotes 'High Risk', maintain an urgent, protective clinical tone.\n"
        "- If you don't know the answer or the context is insufficient, say you don't know.\n\n"
        "CRITICAL: Always append this exact clinical disclaimer at the very end of your response on a new line: "
        "'[DISCLAIMER: This information is derived from verified global health reference materials and a clinical ML risk predictor. It is for educational purposes only. Consult a professional physician for real medical interventions.]'\n\n"
        "Context:\n{context}\n\n"
        "Chat History:\n{history}"
    )),
    ("human", "{input}")
])

def helper_clean_source_name(raw_source):
    base_name = os.path.basename(raw_source).lower()
    if "gale" in base_name or "medical_book" in base_name:
        return "Gale Encyclopedia of Medicine"
    elif "medlineplus" in base_name:
        return "MedlinePlus Clinical Reference"
    elif "who" in base_name:
        return "WHO Global Health Framework"
    return "Verified Medical Reference Manual"

# 1. Root Route
@app.route('/', methods=['GET'])
def home():
    return render_template('chat.html')

# 2. API Endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get("question", "")
    
    # سحب قراءة الـ ML Risk Score القادمة من الفرونت إند (الزرار العائم)
    ml_risk_status = data.get("risk_status", "No instant biometric risk calculated yet.")
    
    # Extract chat history from the frontend payload
    raw_history = data.get("history", [])
    formatted_history = "\n".join([f"User: {msg.get('user', '')}\nBot: {msg.get('bot', '')}" for msg in raw_history])
    
    try:
        retrieved_docs = retriever.invoke(user_question)
        citations = []
        seen_citations = set()
        
        for doc in retrieved_docs:
            raw_src = doc.metadata.get("source", "Unknown Document")
            clean_src = helper_clean_source_name(raw_src)
            raw_page = doc.metadata.get("page_label") or doc.metadata.get("page") or "N/A"
            clean_page = str(int(float(raw_page))) if raw_page != "N/A" else "N/A"
            
            citation_string = f"{clean_src} (Page {clean_page})"
            if citation_string not in seen_citations:
                seen_citations.add(citation_string)
                citations.append({"book": clean_src, "page": clean_page})
        
        # استخراج نصوص مراجع باينكون
        retrieved_docs_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # بناء الـ Context الهجين (المرجع العربي + نتيجة الـ ML + مراجع باينكون)
        full_context = (
            f"{ARABIC_MEDICAL_REFERENCE}\n\n"
            f"=== Patient ML Risk Assessment ===\n{ml_risk_status}\n\n"
            f"=== Retrieved Medical References ===\n{retrieved_docs_text}"
        )
        
        formatted_prompt = prompt.format_messages(
            context=full_context, 
            history=formatted_history, 
            input=user_question
        )
        ai_response = llm.invoke(formatted_prompt).content
        
        # Strict Citation Purging: If Gemini admits it doesn't know, clear the sources
        response_lower = ai_response.lower()
        if "do not know" in response_lower or "don't know" in response_lower:
            citations = []
            
        return jsonify({
            "status": "success",
            "answer": ai_response,
            "citations": citations
        })
    except Exception as e:
        print(f"[Error] {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

# تشغيل السيرفر على البورت 7860 وتفعيل الاستقبال الخارجي على هانجينج فيس
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860, debug=False)
