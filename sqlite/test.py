import streamlit as st
import time
import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 配置信息
DATA_FILE = "./data/processed_data.json"
EMBEDDING_MODEL_NAME = r"G:\shujuwajue\4\exp04-easy-rag-system\exp04-easy-rag-system\all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = r"G:\shujuwajue\4\exp04-easy-rag-system\exp04-easy-rag-system\Qwen2.5-0.5B"
TOP_K = 15  # 要展示的文档数量
FETCH_K = 100  # 从数据库中检索的文档总数
DB_PATH = './data/medical_rag.db'
MIN_ABSTRACT_LENGTH = 300  # 摘要最小长度，用于过滤文档

# 初始化 SQLite 数据库
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            
            title TEXT,
            abstract TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

# 加载数据
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []

# 检查数据库是否为空
def check_db_empty(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    count = cursor.fetchone()[0]
    return count == 0  # 如果为空，返回 True

# 嵌入模型
@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

# 生成模型
@st.cache_resource
def load_generation_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# 插入数据并显示进度
def insert_data(conn, data, embeddings):
    cursor = conn.cursor()
    progress_bar = st.progress(0)  # 初始化进度条
    for i, (entry, embedding) in enumerate(zip(data, embeddings)):
        cursor.execute('''
            INSERT OR REPLACE INTO documents (id, title, abstract, embedding) VALUES (?, ?, ?, ?)
        ''', (entry['id'], entry['title'], entry['abstract'], embedding.tobytes()))
        # 更新进度条
        progress_bar.progress((i + 1) / len(data))
    conn.commit()
    progress_bar.empty()  # 完成后清除进度条

# 嵌入查询文档并计算相似度
def search_similar_documents(conn, query_embedding, top_k=TOP_K, fetch_k=FETCH_K):
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, abstract, embedding FROM documents")
    documents = cursor.fetchall()

    # 计算每个文档嵌入与查询嵌入的相似度（使用余弦相似度）
    scores = []
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)  # 正规化查询嵌入
    for doc in documents:
        # 只考虑摘要长度大于等于 MIN_ABSTRACT_LENGTH 的文档
        if len(doc[2]) < MIN_ABSTRACT_LENGTH:
            continue
        
        doc_embedding = np.frombuffer(doc[3], dtype=np.float32)  # 将 BLOB 还原为数组
        if np.linalg.norm(doc_embedding) == 0:  # 异常情况检查
            continue
        doc_embedding_normalized = doc_embedding / np.linalg.norm(doc_embedding)  # 正规化文档嵌入
        score = np.dot(doc_embedding_normalized, query_embedding_normalized)  # 计算余弦相似度
        scores.append((score, doc))

    # 按相似度排序并选取前 fetch_k 个文档
    scores.sort(reverse=True, key=lambda x: x[0])
    return [(score, doc) for score, doc in scores[:fetch_k]]  # 返回相似度和文档内容

# Streamlit 界面设定
st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 系统 (SQLite)")
st.markdown(f"使用 `{EMBEDDING_MODEL_NAME}` 和 `{GENERATION_MODEL_NAME}`。")

# 初始化和加载模型
embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)

# 初始化数据库
conn = init_db()

# 检查数据库是否为空并加载数据
if check_db_empty(conn):
    pubmed_data = load_data(DATA_FILE)
    if pubmed_data:
        # 生成嵌入
        embeddings = embedding_model.encode([doc['abstract'] for doc in pubmed_data], show_progress_bar=True)
        insert_data(conn, pubmed_data, embeddings)
    else:
        st.warning("加载的数据为空，无法插入到数据库。")

# RAG 交互部分
query = st.text_input("请提出关于已索引医疗文章的问题:", key="query_input")

if st.button("获取答案", key="submit_button") and query:
    start_time = time.time()

    # 对查询进行嵌入
    query_embedding = embedding_model.encode(query)

    # 检索与查询相似的文档
    retrieved_docs_with_scores = search_similar_documents(conn, query_embedding, top_k=TOP_K)

    if not retrieved_docs_with_scores:
        st.warning("在数据库中找不到相关文档。")
    else:
        st.subheader("检索到的相关文档:")
        for score, doc in retrieved_docs_with_scores[:TOP_K]:  # 只显示前TOP_K篇
            with st.expander(doc[1], expanded=False):  # 展开标题以查看全文内容
                st.write(f"**标题:** {doc[1]}")  # 显示标题
                st.write(f"**内容:** {doc[2]}")  # 显示摘要
                st.write(f"**相似度得分:** {score:.4f}")  # 显示相似度得分
        
        # 生成答案并显示进度
        st.subheader("生成的答案:")
        context = "\n\n---\n\n".join([f"标题: {doc[1]}\n内容: {doc[2]}" for _, doc in retrieved_docs_with_scores[:TOP_K]])
        prompt = f"""根据以下上下文生成答案：
        上下文：
        {context}

        用户问题: {query}

        答案：
        """
        
        inputs = tokenizer(prompt, return_tensors="pt").to(generation_model.device)

        # 添加进度条进行生成
        completion_progress_bar = st.progress(0)  # 初始化进度条
        with torch.no_grad():
            outputs = generation_model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
            # 在生成过程中更新进度条
            for i in range(101):
                completion_progress_bar.progress(i)
                time.sleep(0.02)  # 模拟生成时间

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(response)
        completion_progress_bar.empty()  # 完成后清除进度条

    end_time = time.time()
    st.info(f"总耗时: {end_time - start_time:.2f} 秒")

# 关闭数据库连接
conn.close()
