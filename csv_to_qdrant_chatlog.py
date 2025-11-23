import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# === 設定參數 ===
CHATLOG_FOLDER = r"C:\Users\Ching\OneDrive\桌面\阿邱\暨大\必修\專題\graduate_project_git\CSV_chatlog"
COLLECTION_NAME = "chat_history_v1"
DIMENSION = 384

# === 初始化模型與 Qdrant 客戶端 ===
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
qdrant = QdrantClient("localhost", port=32768)

# === 建立 Qdrant Collection 若不存在 ===
collections = qdrant.get_collections().collections
collection_names = [c.name for c in collections]

if COLLECTION_NAME not in collection_names:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE)
    )
    print(f"📌 已建立 Collection：{COLLECTION_NAME}")

# === 轉換 chat_log.csv 並加入 Qdrant ===
point_id = 0

for filename in os.listdir(CHATLOG_FOLDER):
    if filename.endswith(".csv"):
        filepath = os.path.join(CHATLOG_FOLDER, filename)
        print(f"➡ 正在讀取：{filepath}")

        df = pd.read_csv(filepath, encoding="utf-8-sig")

        for _, row in df.iterrows():
            timestamp = str(row.get("timestamp", "")).strip()
            user_question = str(row.get("user_question", "")).strip()
            ai_answer = str(row.get("ai_answer", "")).strip()

            # 跳過空白 question
            if not user_question:
                continue

            # 將 user question encode
            vector = model.encode(user_question).tolist()

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "timestamp": timestamp,
                    "user_question": user_question,
                    "ai_answer": ai_answer
                }
            )

            qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
            print(f"✅ 已加入 Point {point_id}：{user_question[:25]}...")

            point_id += 1

print("🎉 chat_log CSV 已成功轉換並儲存至 Qdrant！")