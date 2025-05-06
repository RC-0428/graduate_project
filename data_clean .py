import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 設定資料夾與模型
# csv_folder：CSV檔案置放資料夾路徑
# output_csv：去重病整合後的新CSV檔名
# model：使用的模型種類
csv_folder = r"C:\Users\Ching\Downloads\CSV"
output_csv = "merged_deduped_output2-7.csv"
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 記錄所有段落
paragraphs = []

# 逐一讀入並處理 CSV 檔案
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_folder, file), encoding="utf-8")
        
        # 組段落（此處預設每列為一段，你可以改為自定段落邏輯）
        for _, row in df.iterrows():
            text = " ".join(row.dropna().astype(str)).strip()
            if text:
                paragraphs.append(text)

print(f"總段落數：{len(paragraphs)}")

# 對所有段落做語意向量嵌入
embeddings = model.encode(paragraphs, convert_to_tensor=True)

# 用 cosine similarity 做語意去重
unique_paragraphs = [] # 存放「不重複的段落文字」
unique_embeddings = [] # 存放「不重複段落」對應的向量

# 用 enumerate 來走訪每一段原始段落與對應向量
for i, (text, emb) in enumerate(zip(paragraphs, embeddings)):
    is_duplicate = False# 預設這段不是重複的

    # 將目前段落的向量與已經保留的向量逐一比對
    for u_emb in unique_embeddings:
        # 使用 cosine similarity 計算語意相似度（越接近 1 越相似）
        # 如果相似度超過 0.9，視為重複
        if util.cos_sim(emb, u_emb) > 0.9: 
            is_duplicate = True # 標記為重複
            break 
     # 如果不是重複的，就加入結果清單中
    if not is_duplicate:
        unique_paragraphs.append(text) # 保存這段文字
        unique_embeddings.append(emb) # 保存其向量

# 儲存結果
df_result = pd.DataFrame({"text": unique_paragraphs})
df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 去重後段落數：{len(unique_paragraphs)}，已儲存為：{output_csv}")
