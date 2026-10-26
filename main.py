import os
import asyncio
import hashlib
import time
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import requests
import chromadb
from chromadb import PersistentClient
import logging
from typing import List


# -------------------------- 日志配置 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------- 加载环境变量 --------------------------
load_dotenv()
import dashscope


# 大模型LLM 火山方舟
llm_client = AsyncOpenAI(
    api_key=os.getenv("VOLC_ARK_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    timeout=30.0,
    max_retries=0
)
# dashscope全局key 用于Rerank
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# -------------------------- 读取.env配置参数 --------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 5))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 3))
SIMILAR_THRESHOLD = float(os.getenv("SIMILAR_THRESHOLD", 0.3))
DEBUG_SIMILARITY = os.getenv("DEBUG_SIMILARITY", "True") == "True"
EMBED_MODEL = os.getenv("EMBED_MODEL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
VOLC_MODEL_EP = os.getenv("VOLC_MODEL_EP")
MAX_CHAT_TURN = int(os.getenv("MAX_CHAT_TURN", 6))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))

EMBED_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_TIMEOUT = 15
RERANK_TIMEOUT = 6
COOLDOWN_SECONDS = 60

# 外部检索接口异常后的熔断时间戳：冷却期内直接走本地关键词降级，避免每次请求都等超时
_embed_failed_at = 0.0
_rerank_failed_at = 0.0


def embed_texts(texts: List[str]) -> List[List[float]]:
    """同步批量向量化，调用方用 asyncio.to_thread 包裹，避免阻塞事件循环。"""
    resp = requests.post(
        f"{EMBED_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}"},
        json={"model": EMBED_MODEL, "input": texts},
        timeout=EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


# -------------------------- Chroma向量库初始化 --------------------------
chroma_client = PersistentClient(path="./chroma_db")


def _get_collection():
    """统一使用 cosine 距离；旧 collection 若为 L2 则重建，避免相似度换算错误。"""
    try:
        col = chroma_client.get_collection("rag_knowledge_base")
        if (col.metadata or {}).get("hnsw:space") == "cosine":
            return col
        logger.warning("检测到旧 collection 非 cosine 距离，重建中")
        chroma_client.delete_collection("rag_knowledge_base")
    except Exception:
        pass
    return chroma_client.create_collection(
        name="rag_knowledge_base", metadata={"hnsw:space": "cosine"}
    )


collection = _get_collection()

# -------------------------- MD5指纹数据库（sqlite）--------------------------
import sqlite3
def init_md5_db():
    conn = sqlite3.connect("md5_record.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS file_md5 (
        file_path TEXT PRIMARY KEY,
        md5_hash TEXT,
        update_time DATETIME
    )''')
    conn.commit()
    conn.close()
    logger.info("md5指纹数据库初始化成功")

def get_file_md5(filepath: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_file_changed(filepath: Path) -> tuple[bool, str]:
    current_md5 = get_file_md5(filepath)
    conn = sqlite3.connect("md5_record.db")
    cur = conn.cursor()
    res = cur.execute("SELECT md5_hash FROM file_md5 WHERE file_path=?", (str(filepath),)).fetchone()
    if res is None:
        conn.close()
        return True, current_md5
    old_md5 = res[0]
    conn.close()
    return old_md5 != current_md5, current_md5

def update_md5_record(filepath: Path, md5_str: str):
    conn = sqlite3.connect("md5_record.db")
    cur = conn.cursor()
    cur.execute('''INSERT OR REPLACE INTO file_md5(file_path,md5_hash,update_time)
        VALUES (?,?,datetime('now'))''', (str(filepath), md5_str))
    conn.commit()
    conn.close()

# -------------------------- 文本切分工具 --------------------------
def split_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# -------------------------- 文档加载 支持txt、pdf --------------------------
import pdfplumber
import re
def load_file_content(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    content = ""
    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    elif suffix == ".pdf":
        text_list = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_list.append(page_text)
        content = "\n".join(text_list)
        # =========新增文本清洗=========
        # 1. 将多个换行符替换为单个换行
        content = re.sub(r"\n+", "\n", content)
        # 2. 将连续多个空格、tab压缩为单个空格
        content = re.sub(r"\s+", " ", content)
        content = content.strip()
        logger.info(f"PDF解析+清洗完成: {file_path}")
    return content


# -------------------------- 批量入库【重点修复：分批Embedding，单批最多25】 --------------------------
async def add_folder_to_rag(folder_path: str = "./data"):
    """扫描目录做增量入库，返回本次新增 / 跳过 / 失败的文件清单。"""
    stats = {"added": [], "skipped": [], "failed": []}
    data_dir = Path(folder_path)
    if not data_dir.exists():
        logger.warning(f"文件夹{folder_path}不存在")
        return stats
    file_list = []
    for f in data_dir.iterdir():
        if f.is_file() and f.suffix.lower() in [".txt", ".pdf"]:
            file_list.append(f)
    logger.info(f"====扫描文件夹:{folder_path}，发现文件数量:{len(file_list)} ====")
    BATCH_SIZE_EMB = 10  # DashScope embedding 单批上限较小，10 条一批更稳
    # 向量库为空时强制重建，避免 MD5 记录存在但 collection 被清空后跳过入库
    force_rebuild = collection.count() == 0
    for fp in file_list:
        changed, new_md5 = check_file_changed(fp)
        if not changed and not force_rebuild:
            logger.info(f"✅ {fp.name} 文件未修改，跳过处理")
            stats["skipped"].append(fp.name)
            continue
        logger.info(f"🔄 处理文件: {fp.name}")
        text = load_file_content(fp)
        if not text.strip():
            logger.warning(f"{fp.name} 读取内容为空，跳过")
            stats["failed"].append(fp.name)
            update_md5_record(fp, new_md5)
            continue
        chunks = split_text(text)
        logger.info(f"{fp.name} 切片完成，chunk数量:{len(chunks)}")

        # =====分批调用Embedding=====
        try:
            all_embeddings = []
            for batch_idx in range(0, len(chunks), BATCH_SIZE_EMB):
                batch_chunks = chunks[batch_idx: batch_idx + BATCH_SIZE_EMB]
                all_embeddings.extend(await asyncio.to_thread(embed_texts, batch_chunks))
        except Exception as e:
            logger.error(f"{fp.name} 向量化失败，跳过该文件：{e}")
            stats["failed"].append(fp.name)
            continue

        ids = [f"{fp.name}:{i}" for i in range(len(chunks))]
        metadatas = [{"source": fp.name, "chunk_index": i} for i in range(len(chunks))]

        # 先删同源旧片段再写入，避免文件更新后残留脏数据
        try:
            collection.delete(where={"source": fp.name})
        except Exception as e:
            logger.warning(f"清理 {fp.name} 旧片段失败：{e}")

        # 用 upsert 而不是 add：同一 id 重复写入不再抛 DuplicateIDError
        collection.upsert(
            documents=chunks,
            embeddings=all_embeddings,
            ids=ids,
            metadatas=metadatas
        )
        update_md5_record(fp, new_md5)
        stats["added"].append(fp.name)
        logger.info(f"✅ {fp.name} 入库完成")
    return stats

# -------------------------- 检索 + Rerank【修复：使用dashscope原生Rerank】 --------------------------
def _query_grams(text: str) -> set:
    """中文按字符 bigram、英文/数字按词切分，用于本地关键词降级检索。"""
    grams = set()
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            grams.add(seg)
        grams.update(seg[i:i + 2] for i in range(len(seg) - 1))
    grams.update(w.lower() for w in re.findall(r"[A-Za-z0-9]+", text))
    return grams


def _keyword_search(query: str) -> List[str]:
    """Embedding 不可用时的降级检索：在本地 Chroma 语料上按 bigram 重叠度打分。"""
    try:
        data = collection.get(include=["documents"])
    except Exception as e:
        logger.error(f"关键词降级检索失败：{e}")
        return []

    docs = data.get("documents") or []
    q_grams = _query_grams(query)
    if not q_grams:
        return []

    scored = []
    for doc in docs:
        overlap = len(q_grams & _query_grams(doc))
        if overlap:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    result = [doc for _, doc in scored[:RERANK_TOP_K]]
    logger.info(f"关键词降级检索命中 {len(result)} 条")
    return result


async def _rerank(query: str, docs: List[str]) -> List[str]:
    """Rerank 失败或超时时降级为向量召回顺序。"""
    global _rerank_failed_at
    if time.time() - _rerank_failed_at < COOLDOWN_SECONDS:
        return docs[:RERANK_TOP_K]

    try:
        from dashscope import TextReRank
        rerank_resp = await asyncio.wait_for(
            asyncio.to_thread(
                TextReRank.call,
                model=RERANK_MODEL,
                query=query,
                documents=docs,
            ),
            timeout=RERANK_TIMEOUT,
        )
        if rerank_resp.status_code != 200:
            _rerank_failed_at = time.time()
            logger.error(f"Rerank接口调用失败，降级使用原始召回结果, err:{rerank_resp.message}")
            return docs[:RERANK_TOP_K]

        rerank_results = sorted(rerank_resp.output["results"], key=lambda x: x["relevance_score"], reverse=True)
        reranked_docs = []
        for item in rerank_results[:RERANK_TOP_K]:
            score = item["relevance_score"]
            # ✅重点：用index从原始docs取出原文！
            doc_text = docs[item["index"]] or ""
            reranked_docs.append(doc_text)
            logger.info(f"Rerank 得分:{score:.3f} 文本片段: {doc_text[:60]}...")
        logger.info(f"Rerank重排完成，最终选取{len(reranked_docs)}条片段送入LLM")
        return reranked_docs

    except Exception as e:
        _rerank_failed_at = time.time()
        logger.error(f"Rerank 异常，{COOLDOWN_SECONDS}s 内降级使用向量召回结果：{e}")
        return docs[:RERANK_TOP_K]


async def retrieve_knowledge(query: str) -> List[str]:
    """向量召回 + 阈值过滤 + Rerank；外部接口异常时熔断并降级为本地关键词检索。"""
    global _embed_failed_at
    query = (query or "").strip()
    if not query:
        return []

    if time.time() - _embed_failed_at < COOLDOWN_SECONDS:
        return _keyword_search(query)

    try:
        query_emb = (await asyncio.to_thread(embed_texts, [query]))[0]
        _embed_failed_at = 0.0
        logger.info(f"当前查询向量维度:{len(query_emb)}")
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=RETRIEVE_TOP_K
        )
        docs = results["documents"][0]
        distances = results["distances"][0]
        filtered_docs = []
        for doc, dist in zip(docs, distances):
            # collection 使用 cosine 空间，distance = 1 - cosine_similarity
            sim = max(0.0, 1 - dist)
            if DEBUG_SIMILARITY:
                logger.info(f"片段相似度={sim:.3f}, cosine距离={dist:.3f}")
            if sim >= SIMILAR_THRESHOLD:
                filtered_docs.append(doc)
        logger.info(f"向量召回+阈值过滤后片段数量: {len(filtered_docs)}")
        if not filtered_docs:
            return _keyword_search(query)

        return await _rerank(query, filtered_docs)

    except (requests.RequestException, asyncio.TimeoutError) as e:
        _embed_failed_at = time.time()
        logger.warning(f"Embedding 接口超时，{COOLDOWN_SECONDS}s 内改用关键词检索：{e}")
        return _keyword_search(query)
    except Exception as e:
        logger.error(f"检索知识库未知异常: {str(e)}", exc_info=True)
        return _keyword_search(query)




# -------------------------- 会话持久化SQLite --------------------------
def init_chat_db():
    conn = sqlite3.connect("chat_session.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS chat_msg (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()
    logger.info("会话数据库初始化成功")

def load_chat_history(session_id: str) -> List[dict]:
    conn = sqlite3.connect("chat_session.db")
    cur = conn.cursor()
    rows = cur.execute("SELECT role,content FROM chat_msg WHERE session_id=? ORDER BY id",(session_id,)).fetchall()
    conn.close()
    msgs = []
    for r in rows:
        msgs.append({"role":r[0], "content":r[1]})
    return msgs

def save_chat_msg(session_id: str, role: str, content: str):
    conn = sqlite3.connect("chat_session.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO chat_msg(session_id,role,content) VALUES (?,?,?)",(session_id,role,content))
    conn.commit()
    conn.close()

# -------------------------- 流式问答 --------------------------
async def chat_stream(session_id: str, user_query: str):
    # 检索知识库
    ref_docs = await retrieve_knowledge(user_query)
    logger.info(f"【调试-检索返回文档列表】\n{ref_docs}")
    ref_context = "\n\n".join(ref_docs)
    logger.info(f"【调试-拼接后的参考上下文】\n{ref_context}")
    history = load_chat_history(session_id)
    # 构建prompt
    system_prompt = f"""你是知识库问答助手，基于下面参考资料回答用户问题。
参考资料：
{ref_context}
你只能严格参考下面提供的参考资料回答问题，禁止自己额外扩写、发挥。
如果参考资料里有答案，直接引用原文回答，不要改写；
如果资料没有相关内容，直接回答没有找到资料。
"""
    messages = [{"role":"system", "content":system_prompt}]
    messages.extend(history[-MAX_CHAT_TURN:])
    messages.append({"role":"user", "content":user_query})
    # 调用火山LLM流式输出
    stream = await llm_client.chat.completions.create(
        model=VOLC_MODEL_EP,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        stream=True
    )
    full_answer = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_answer += text
            yield text
    # 保存对话
    save_chat_msg(session_id, "user", user_query)
    save_chat_msg(session_id, "assistant", full_answer)

# -------------------------- FastAPI接口 --------------------------
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


# -------------------------- 启动预热 --------------------------
async def warmup_services():
    """后台预热：提前跑一次检索链路并建立 LLM 端点连接，让首个用户请求不再承担冷启动开销。"""
    t0 = time.time()
    try:
        docs = await retrieve_knowledge("预热检索")
        logger.info(f"检索链路预热完成，命中 {len(docs)} 条，耗时 {time.time() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"检索预热失败，不影响启动：{e}")

    t1 = time.time()
    try:
        await llm_client.chat.completions.create(
            model=VOLC_MODEL_EP,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        logger.info(f"LLM 端点预热完成，耗时 {time.time() - t1:.2f}s")
    except Exception as e:
        logger.warning(f"LLM 预热失败，不影响启动：{e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """uvicorn 启动时初始化数据库并增量同步知识库。"""
    init_md5_db()
    init_chat_db()
    await add_folder_to_rag("./data")
    # 放到后台任务里，预热耗时不能拖慢服务启动
    app.state.warmup_task = asyncio.create_task(warmup_services())
    yield


app = FastAPI(title="RAG知识库服务", lifespan=lifespan)


@app.get("/")
async def index_page():
    return FileResponse(Path(__file__).parent / "index.html")


class QueryReq(BaseModel):
    session_id: str
    question: str

@app.post("/rag/query")
async def rag_api(req: QueryReq):
    resp_text = ""
    async for seg in chat_stream(req.session_id, req.question):
        resp_text += seg
    return {"answer": resp_text}
from fastapi import UploadFile, File
import shutil
from pathlib import Path

DATA_FOLDER = Path("./data")
DATA_FOLDER.mkdir(exist_ok=True)

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    allow_suffix = {".txt", ".pdf"}
    filename = file.filename
    file_path = DATA_FOLDER / filename
    suffix = Path(filename).suffix.lower()
    if suffix not in allow_suffix:
        return {"code": 400, "msg": f"不支持文件类型，仅支持txt、pdf，当前:{suffix}"}

    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 调用增量入库函数，按实际处理结果返回提示
    stats = await add_folder_to_rag("./data")

    if filename in stats["added"]:
        return {"code": 200, "status": "added",
                "msg": f"{filename} 已切分并写入向量库",
                "saved_path": str(file_path)}
    if filename in stats["skipped"]:
        return {"code": 200, "status": "skipped",
                "msg": f"{filename} 内容未变化（MD5 一致），已跳过重复入库",
                "saved_path": str(file_path)}
    if filename in stats["failed"]:
        return {"code": 500, "status": "failed",
                "msg": f"{filename} 处理失败，请检查文件内容或向量化服务",
                "saved_path": str(file_path)}
    return {"code": 200, "status": "pending",
            "msg": f"{filename} 已保存，入库未触发变更",
            "saved_path": str(file_path)}


# -------------------------- 主入口 --------------------------
async def main():
    init_md5_db()
    init_chat_db()
    logger.info("=====工程化RAG【增量更新版本】启动 =====")
    # 第一次入库打开，入库完成注释
    await add_folder_to_rag("./data")

    # 交互式问答
    session_id = "test_session_001"
    while True:
        user_input = input("\n请输入问题(exit退出): ")
        if user_input.strip() == "exit":
            break
        print("AI: ", end="", flush=True)
        async for chunk in chat_stream(session_id, user_input):
            print(chunk, end="", flush=True)
        print("")

if __name__ == "__main__":
    asyncio.run(main())
