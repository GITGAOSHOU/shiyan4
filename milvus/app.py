import streamlit as st
import time
import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache' 

# 更新后的配置
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT,  # 添加新的配置项
    id_to_doc_map, INDEX_PARAMS, SEARCH_PARAMS, VECTOR_DIM  # 确保新增参数被导入
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
from rag_core import generate_answer

# --- 初始化Milvus连接 ---
def get_milvus_client():
    try:
        connections.connect(host="localhost", port="19530")
        return True
    except Exception as e:
        st.error(f"连接Milvus失败: {str(e)}")
        return False

# --- 创建集合 ---
def setup_milvus_collection():
    try:
        # 检查集合是否存在
        if not utility.has_collection(COLLECTION_NAME):
            # 1. 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
            ]
            
            # 2. 创建集合
            schema = CollectionSchema(fields, description="医疗文档检索集合")
            collection = Collection(COLLECTION_NAME, schema)
            
            # 3. 创建索引（关键修改点）
            index_params = {
                "index_type": INDEX_PARAMS["index_type"],
                "metric_type": INDEX_PARAMS["metric_type"],
                "params": INDEX_PARAMS["params"]
            }
            collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            
            # 4. 确保索引创建完成
            if len(collection.indexes) == 0:
                raise RuntimeError("索引创建失败")
                
            st.success(f"集合 {COLLECTION_NAME} 创建成功")
            return True
            
        else:
            # 处理已存在集合的情况
            collection = Collection(COLLECTION_NAME)
            
            # 检查索引是否存在
            if len(collection.indexes) == 0:
                st.error(f"集合 {COLLECTION_NAME} 存在但没有索引")
                utility.drop_collection(COLLECTION_NAME)
                return setup_milvus_collection()  # 递归调用
            
            collection.load()
            st.success(f"成功加载集合: {COLLECTION_NAME}")
            return True
            
    except Exception as e:
        st.error(f"操作失败: {str(e)}")
        return False

# --- 索引数据 ---
def index_data_if_needed(data, embedding_model):
    collection = Collection(COLLECTION_NAME)
    if collection.num_entities == 0:
        st.info("开始索引数据...")
        batch_size = 100
        total = min(MAX_ARTICLES_TO_INDEX, len(data))
        
        for i in range(0, total, batch_size):
            batch = data[i:i+batch_size]
            texts = [f"{doc['title']} {doc['abstract']}" for doc in batch]
            
            # 生成向量
            embeddings = embedding_model.encode(texts, show_progress_bar=False)
            
            # 准备插入数据
            entities = [
                [doc['title'] for doc in batch],
                [doc['abstract'] for doc in batch],
                embeddings.tolist()
            ]
            
            # 插入数据
            mr = collection.insert(entities)
            
            # 维护ID映射
            for doc_id, doc in zip(mr.primary_keys, batch):
                id_to_doc_map[doc_id] = doc
            
        collection.flush()
        st.success(f"成功插入 {total} 篇文档")
        return True
    return True

# --- 搜索文档 ---
def search_similar_documents(query, embedding_model, top_k=TOP_K):
    collection = Collection(COLLECTION_NAME)
    collection.load()
    
    # 生成查询向量
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # 执行搜索
    search_params = SEARCH_PARAMS
    results = collection.search(
        data=[query_embedding],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["title", "abstract"]
    )
    
    # 处理结果
    ids = []
    distances = []
    docs = []
    for hits in results:
        for hit in hits:
            ids.append(hit.id)
            distances.append(hit.distance)
            docs.append({
                "title": hit.entity.get("title"),
                "abstract": hit.entity.get("abstract")
            })
    return ids, distances, docs

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📄 医疗 RAG 系统 (Milvus Standalone)")
st.markdown(f"使用 Milvus Standalone, `{EMBEDDING_MODEL_NAME}`, 和 `{GENERATION_MODEL_NAME}`。")

# 初始化连接
if get_milvus_client():
    if setup_milvus_collection():
        # 加载模型
        embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
        generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)
        
        if embedding_model and generation_model and tokenizer:
            pubmed_data = load_data(DATA_FILE)
            
            if pubmed_data:
                indexing_successful = index_data_if_needed(pubmed_data, embedding_model)
                
                st.divider()
                
                if indexing_successful:
                    query = st.text_input("请提出关于已索引医疗文章的问题:", key="query_input")
                    
                    if st.button("获取答案") and query:
                        start_time = time.time()
                        
                        with st.spinner("正在搜索相关文档..."):
                            ids, distances, docs = search_similar_documents(query, embedding_model)
                        
                        if not ids:
                            st.warning("未找到相关文档")
                        else:
                            st.subheader("检索到的上下文文档:")
                            for i, doc in enumerate(docs):
                                with st.expander(f"文档 {i+1} (距离: {distances[i]:.4f}) - {doc['title'][:60]}"):
                                    st.write(f"**标题:** {doc['title']}")
                                    st.write(f"**摘要:** {doc['abstract']}")
                            
                            st.divider()
                            
                            with st.spinner("正在生成答案..."):
                                answer = generate_answer(query, docs, generation_model, tokenizer)
                                st.subheader("生成的答案:")
                                st.write(answer)
                            
                            st.info(f"总耗时: {time.time()-start_time:.2f}秒")

# 侧边栏信息
st.sidebar.header("系统配置")
st.sidebar.markdown(f"**Milvus 地址:** {MILVUS_HOST}:{MILVUS_PORT}")
st.sidebar.markdown(f"**向量维度:** {VECTOR_DIM}")
st.sidebar.markdown(f"**索引参数:** `{INDEX_PARAMS}`")
st.sidebar.markdown(f"**搜索参数:** `{SEARCH_PARAMS}`")