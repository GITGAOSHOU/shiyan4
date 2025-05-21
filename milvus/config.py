MILVUS_HOST = "localhost"        # Milvus服务地址
MILVUS_PORT = "19530"            # Milvus服务端口
COLLECTION_NAME = "medical_rag"  # 集合名称
VECTOR_DIM = 384                 # 必须与嵌入模型维度匹配（all-MiniLM-L6-v2的维度是384）

# 数据配置
DATA_FILE = "./data/processed_data.json"

# 模型配置
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# 索引与搜索参数
MAX_ARTICLES_TO_INDEX = 500      # 最大索引文章数
TOP_K = 3                        # 检索数量

# 索引配置（IVF_FLAT参数）
INDEX_PARAMS = {
    "index_type": "IVF_FLAT",    # 明确指定索引类型
    "metric_type": "L2",         # 单独指定度量类型
    "params": {"nlist": 128}     # IVF 特定参数
}

# 搜索参数
SEARCH_PARAMS = {
    "metric_type": "L2",         # 需要与索引的metric_type一致
    "params": {"nprobe": 16}     # 搜索的聚类单元数
}

# 生成参数
MAX_NEW_TOKENS_GEN = 512         # 生成最大token数
TEMPERATURE = 0.7                # 温度参数
TOP_P = 0.9                      # 核心采样率
REPETITION_PENALTY = 1.1         # 重复惩罚系数

# （可选）文档映射（如果从Milvus直接获取元数据可删除）
id_to_doc_map = {}