import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# 設定參數
# CSV_FOLDER：CSV檔案置放資料夾路徑
# COLLECTION_NAME：向量資料庫名稱
# DIMENSION：使用的模型向量維度
CSV_FOLDER = r"C:\Users\Ching\OneDrive\桌面\阿邱\暨大\必修\專題\graduate_project_git\CSV_v3"
COLLECTION_NAME = "my_documents2-17v1"
DIMENSION = 384 

# 初始化模型與 Qdrant 客戶端 
# model：使用的模型種類
# qdrant：qdrant連接的位址以及port
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
qdrant = QdrantClient("localhost", port=32768)

# 建立 Collection（若尚未建立）
if COLLECTION_NAME not in qdrant.get_collections().collections:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE)
    )

# 處理 CSV 檔案並加入向量 
point_id = 0  # 每筆資料要有一個唯一的ID，從0開始
# 逐一讀入並處理 CSV 檔案
for filename in os.listdir(CSV_FOLDER):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(CSV_FOLDER, filename), encoding="utf-8")
        for _, row in df.iterrows():
            chunk_text = " ".join(row.dropna().astype(str)) # 將這一行的所有欄位合併成一段文字（去除空值）
            if chunk_text.strip(): # 如果這段文字不是空的才處理
                vector = model.encode(chunk_text).tolist()  # 使用語意模型將文字轉換成向量
                # 建立一筆 Qdrant 的資料點（包含 ID、向量、原始文字）
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"chunk_text": chunk_text}
                )
                # 將這筆資料加入 Qdrant 的向量資料庫中
                qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
                # ID 加 1，準備處理下一筆
                point_id += 1

print("✅ CSV 資料已成功轉換並儲存至 Qdrant！")
