from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient("localhost", port=32768)

client.delete(
    collection_name="chat_history_v2",
    points_selector=models.FilterSelector(
        filter=models.Filter(must=[])  # must=[] 表示「全部」
    )
)

print("已清空所有 points（但保留 collection）")
