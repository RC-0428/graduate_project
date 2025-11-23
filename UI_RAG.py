from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import gradio as gr
import requests
import os
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import threading
import csv
from datetime import datetime

# 讀取環境變數（或直接貼上你的 Token 與 Secret）
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "fMbCvIudIk+Qzafcx2N8QvOkb/2rmSdw+wWTwdX7zhzz7dndEuGooi4YljZOi304Bek7QghN0qp6hMZy5Zuhqjzhc4+OUSdydqevK/YO7G8OIwLZ1Ya+eWAbg1sdhNNtykvKokCdYLcSPmHx3rt2ewdB04t89/1O/w1cDnyilFU=")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "113f99564b7941732e96c4fd1debf395")

app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Webhook 路徑，LINE 會把使用者訊息 POST 到這裡
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

# 接收文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    answer = chat(user_text)  # 呼叫你的 LLM 問答系統
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer)
    )

# 向量搜尋器，設定相關參數
class QdrantSearcher:
    # collection_name：欲使用的向量資料庫名稱
    # self.client：qdrant連接的位址以及port
    # self.model：使用的模型種類
    def __init__(self, collection_name="my_documents2-17v1"):
        self.client = QdrantClient("localhost", port=32768)
        self.collection_name = collection_name
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 先查詢典型字典    
    def search_faq(self, user_question: str, limit: int = 1):
        vector = self.model.encode(user_question).tolist()

        results = self.client.search(
            collection_name="QAdic_HV3",
            query_vector=vector,
            limit=limit
        )
        print("⚡ FAQ 搜尋結果（raw）:", results)
        if not results:
            print("❗ FAQ 沒有任何結果，可能 collection 為空")
            return None
        best = results[0]
        print("⚡ FAQ payload:", best.payload)
        print("⚡ FAQ score:", best.score)
        if best.score < 0.60:
            print("❗ score 太低 → 視為不相關")
            return None
        answer = best.payload.get("answer")
        print("⚡ FAQ answer 欄位:", answer)

        return answer
    
    # 再查詢chat_log   
    def search_chatlog(self, user_question: str, limit: int = 1):
        vector = self.model.encode(user_question).tolist()

        results = self.client.search(
            collection_name="chat_history_v2",
            query_vector=vector,
            limit=limit
        )
        print("✒️ ChatLog 搜尋結果（raw）:", results)
        if not results:
            print("❗ ChatLog 沒有任何結果，可能 collection 為空")
            return None
        best = results[0]
        print("✒️ ChatLog payload:", best.payload)
        print("✒️ ChatLog score:", best.score)
        if best.score < 0.60:
            print("❗ score 太低 → 視為不相關")
            return None
        answer = best.payload.get("ai_answer")
        print("✒️ ChatLog answer 欄位:", answer)

        return answer

    # 從原文章搜尋相關資料
    # 定義一個搜尋方法，輸入一個問題（query），回傳語意上最相關的幾段資料
    def search(self, query: str, limit: int = 3):
        vector = self.model.encode(query).tolist() # 把使用者的問題轉成向量格式
        # 向 Qdrant 發送搜尋請求
        results = self.client.search( 
            collection_name=self.collection_name, # 指定要在哪個 collection 裡搜尋
            query_vector=vector,  # 使用轉換後的向量作為搜尋關鍵
            limit=limit # 最多回傳幾筆結果（預設是 3 筆）
        )
        # 從搜尋結果中取出 payload（原始段落文字），組成清單回傳
        return [r.payload.get("chunk_text", "") for r in results]

# 呼叫 LM Studio 模型並取得模型回答
def ask_lmstudio(context: str, question: str) -> str:
    # url：LM Studio 的 API 端點（預設是本地的 OpenAI 接口）
    url = "http://127.0.0.1:1234/v1/chat/completions"
    # 設定要送給模型的資料（payload）
    payload = {
        "model": "breeze-7b-instruct-v1_0", # 使用的模型種類
        # 模仿 OpenAI 的 Chat API 格式
        "messages": [
            ##{"role": "system", "content": "你是繁體中文知識助手，請根據參考內容並自行補充合理內容來回答使用者問題。"},# 系統提示：設定 AI 的角色與語言
            ##{"role": "user", "content": f"以下是參考內容：\n{context}\n\n問題：{question}"} # 使用者輸入的上下文與問題
            {"role": "system", "content": "你是繁體中文知識助手，請根據參考內容以大約50~80字內簡短回答問題，在回答時必須完圈依照參考內容，請不要擅加資訊"},
            {"role": "user", "content": f"以下是根據資料庫查詢到的參考答案：\n{context}\n\n請根據此參考內容與你的理解來回答使用者的問題：{question}"}
        ],
        "temperature": 0.7,# 控制回答的創造力（0 越穩定，1 越有創意）
        "stream": False  # 是否開啟串流回答（這裡關閉，直接回傳整段）
    }
    try:
        response = requests.post(url, json=payload, timeout=600) # 將資料（payload）透過 HTTP POST 傳給 LM Studio 的 API
        response.raise_for_status() # 如果發生 HTTP 錯誤（例如 404、500）會丟出例外
        result = response.json() # 解析回傳的 JSON 結果
        return result["choices"][0]["message"]["content"].strip() # 從 JSON 裡取出模型的回答內容並移除多餘空白
    
    # 如果發生錯誤（如連不到模型或資料格式錯誤），回傳錯誤訊息
    except Exception as e:
        return f"❌ 發生錯誤：{e}"

# 整合搜尋與回答 
searcher = QdrantSearcher()

#儲存使用者問題&答案的CSV
def save_chat_log_csv(user_question: str, ai_answer: str, filename="CSV_chatlog/chat_log2.csv"):
    # 自動偵測資料夾不存在 → 新增資料夾
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)

        # 如果檔案不存在 → 寫入欄位名稱
        if not file_exists:
            writer.writerow(["timestamp", "user_question", "ai_answer"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_question,
            ai_answer
        ])

#儲存CSV後直接匯入chat_history_v1
def insert_chat_to_qdrant(user_question, ai_answer, timestamp):
    qdrant = QdrantClient("localhost", port=32768)
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    COLLECTION_NAME = "chat_history_v2"
    DIMENSION = 384

    # 若 collection 不存在就先建立
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{COLLECTION_NAME}' 已建立。")

    # 產生向量
    vector = model.encode(user_question).tolist()
    # 自動取得新的 point_id（使用 Qdrant 自動遞增）
    point_id = qdrant.count(collection_name=COLLECTION_NAME).count

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
    print(f"📌 已寫入 Qdrant：{point_id} / {user_question[:20]}...{ai_answer[:20]}...")

# 定義主要的對話函式 chat，輸入問題 query，回傳模型的回答
def chat(query):
    faq_answer = searcher.search_faq(query)  # 查詢典型問答字典
    related_paragraphs = searcher.search(query)  # 查詢原始段落
    chatlog_answer = searcher.search_chatlog(query) #查chatlog
    print("📌 搜尋到的 FAQ：", faq_answer) 
    print("📌 搜尋到的原始段落：", related_paragraphs)
    print("📌 搜尋到的 ChatLog：", chatlog_answer)
    ##results = searcher.search(query) # 先用 Qdrant 向量資料庫搜尋相關段落
    ##if not results:
    if not faq_answer and not related_paragraphs and not chatlog_answer:
        return "❌ 找不到相關內容。請換個說法。" # 如果沒找到內容就回傳提示
    # 組合參考內容
    combined_context = ""
    if faq_answer:
        combined_context += f"【典型問答】\n{faq_answer}\n\n"
    if related_paragraphs:
        combined_context += "【相關段落】\n" + "\n---\n".join(related_paragraphs)
    if chatlog_answer:
        combined_context += f"【典型問答】\n{chatlog_answer}\n\n"
    ##context = "\n\n".join(results)  # 將多個段落用換行分隔組成上下文
    answer = ask_lmstudio(combined_context, query)  # 把上下文與問題一起送去給 LM Studio 生成回答

    save_chat_log_csv(query, answer)      # ★ 在這裡記錄進 chat_log.csv

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insert_chat_to_qdrant(query, answer, timestamp) #將新增的csv內容也新增近qdrant

    return answer

# 建立 Gradio UI 
iface = gr.Interface(
    fn=chat,  # 指定主函式 chat 當作輸入輸出的處理函數
    inputs=gr.Textbox(lines=2, placeholder="請輸入你的問題..."), # 建立一個輸入框，讓使用者輸入問題
    outputs=gr.Textbox(label="AI 回答"), # 建立一個輸出框顯示 AI 的回答
    title="📚 文件問答助理 (Qdrant + LM Studio)",  # 網頁的標題
    description="輸入問題，系統會先從 Qdrant 向量資料庫中搜尋相關段落，再請本地模型回答。" # 介面的簡短說明
)

## 啟動 Gradio 介面，開啟本地瀏覽器讓使用者互動
##iface.launch()

# ============= 同時啟動 Flask + Gradio =============
if __name__ == "__main__":
    def run_gradio():
        iface.launch(server_port=7860, share=True)  # Gradio 在 7860 port

    t = threading.Thread(target=run_gradio)
    t.start()

    app.run(host="0.0.0.0", port=5000)  # Flask 在 5000 port