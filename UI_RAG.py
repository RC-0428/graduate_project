
import gradio as gr
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 向量搜尋器，設定相關參數
class QdrantSearcher:
    # collection_name：欲使用的向量資料庫名稱
    # self.client：qdrant連接的位址以及port
    # self.model：使用的模型種類
    def __init__(self, collection_name="my_documents2-7v2"):
        self.client = QdrantClient("localhost", port=32768)
        self.collection_name = collection_name
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

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
        "model": "yi-1.5-6b-chat", # 使用的模型種類
        # 模仿 OpenAI 的 Chat API 格式
        "messages": [
            {"role": "system", "content": "你是繁體中文知識助手，請根據提供的內容回答問題。"},# 系統提示：設定 AI 的角色與語言
            {"role": "user", "content": f"以下是參考內容：\n{context}\n\n問題：{question}"} # 使用者輸入的上下文與問題
        ],
        "temperature": 0.7,# 控制回答的創造力（0 越穩定，1 越有創意）
        "stream": False  # 是否開啟串流回答（這裡關閉，直接回傳整段）
    }
    try:
        response = requests.post(url, json=payload, timeout=120) # 將資料（payload）透過 HTTP POST 傳給 LM Studio 的 API
        response.raise_for_status() # 如果發生 HTTP 錯誤（例如 404、500）會丟出例外
        result = response.json() # 解析回傳的 JSON 結果
        return result["choices"][0]["message"]["content"].strip() # 從 JSON 裡取出模型的回答內容並移除多餘空白
    
    # 如果發生錯誤（如連不到模型或資料格式錯誤），回傳錯誤訊息
    except Exception as e:
        return f"❌ 發生錯誤：{e}"

# 整合搜尋與回答 
searcher = QdrantSearcher()


# 定義主要的對話函式 chat，輸入問題 query，回傳模型的回答
def chat(query):
    results = searcher.search(query) # 先用 Qdrant 向量資料庫搜尋相關段落
    if not results:
        return "❌ 找不到相關內容。請換個說法。" # 如果沒找到內容就回傳提示
    context = "\n\n".join(results)  # 將多個段落用換行分隔組成上下文
    answer = ask_lmstudio(context, query)  # 把上下文與問題一起送去給 LM Studio 生成回答
    return answer

# 建立 Gradio UI 
iface = gr.Interface(
    fn=chat,  # 指定主函式 chat 當作輸入輸出的處理函數
    inputs=gr.Textbox(lines=2, placeholder="請輸入你的問題..."), # 建立一個輸入框，讓使用者輸入問題
    outputs=gr.Textbox(label="AI 回答"), # 建立一個輸出框顯示 AI 的回答
    title="📚 文件問答助理 (Qdrant + LM Studio)",  # 網頁的標題
    description="輸入問題，系統會先從 Qdrant 向量資料庫中搜尋相關段落，再請本地模型回答。" # 介面的簡短說明
)

# 啟動 Gradio 介面，開啟本地瀏覽器讓使用者互動
iface.launch()
