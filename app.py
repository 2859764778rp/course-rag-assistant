import os
import re
import html
import shutil
import time
import gradio as gr


_FACTUAL_RE = re.compile(r"[0-9]|多少|几|最新|今天|明天|数据|新闻|发展|趋势|前景|进展|原因|为什么|怎么|如何|什么|哪|数量")


def _is_factual_question(q: str) -> bool:
    return bool(_FACTUAL_RE.search(q)) and ("?" in q or "？" in q)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

USER_ID = "19836701050"
GREETING = "同学您好！我是近屿智能课程咨询助手，可以为您解答课程报名、学习安排、文件解析等问题。"


_BACKEND = None


def get_backend():
    global _BACKEND
    if _BACKEND is None:
        from embedding import make_client
        from vectorstore import VectorStore
        from retriever import Retriever
        from qa import QAEngine
        from router import IntentRouter
        from recommend import QuestionRecommender
        from memory import SessionStore
        from parser import process_upload

        emb = make_client()
        vs = VectorStore(emb)
        retriever = Retriever(vs)
        _BACKEND = {
            "qa": QAEngine(retriever),
            "router": IntentRouter(),
            "recommender": QuestionRecommender(),
            "store": SessionStore(),
            "vs": vs,
            "process_upload": process_upload,
        }
    return _BACKEND



_GROUP_ORDER = ["今天", "昨天", "前7天", "前30天", "更早"]


def _group_of(updated_at: float) -> str:
    t = time.localtime(updated_at)
    now = time.localtime()
    day = lambda x: (x.tm_year, x.tm_mon, x.tm_mday)
    d_today = day(now)
    if day(t) == d_today:
        return "今天"
    y = time.localtime(updated_at - 86400)
    if day(y) == d_today:
        return "昨天"
    if time.time() - updated_at < 7 * 86400:
        return "前7天"
    if time.time() - updated_at < 30 * 86400:
        return "前30天"
    return "更早"


def _fmt_label(updated_at: float) -> str:
    t = time.localtime(updated_at)
    if _group_of(updated_at) == "今天":
        return f"{t.tm_hour:02d}:{t.tm_min:02d}"
    return f"{t.tm_mon}/{t.tm_mday}"


def session_groups(user_id: str) -> dict[str, list[tuple[str, str]]]:
    """按时间分组返回会话：{组名: [(显示标签, chat_id), ...]}，组内按 updated_at 降序。"""
    store = get_backend()["store"]
    sessions = store.list_sessions(user_id)  
    grouped: dict[str, list] = {g: [] for g in _GROUP_ORDER}
    for s in sessions:
        grouped[_group_of(s["updated_at"])].append(
            (f"{_fmt_label(s['updated_at'])} · {s['title']}", s["chat_id"])
        )
    return grouped


def _radio_states(user_id: str, current_chat_id: str | None):
    """单个隐藏 Radio：所有会话按 updated_at 降序，仅用于承载选中值。"""
    grouped = session_groups(user_id)
    items = [(label, cid) for g in _GROUP_ORDER for label, cid in grouped[g]]
    val = current_chat_id if current_chat_id in [cid for _, cid in items] else None
    return gr.update(choices=items, value=val)


def render_sidebar_html(user_id: str, current_chat_id: str | None) -> str:
    """按真实日期动态渲染侧边栏分组：每天一个框，框内会话竖向排列。"""
    store = get_backend()["store"]
    sessions = store.list_sessions(user_id)  
    from collections import OrderedDict
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for s in sessions:
        t = time.localtime(s["updated_at"])
        groups.setdefault((t.tm_year, t.tm_mon, t.tm_mday), []).append(s)

    now = time.localtime()
    today = (now.tm_year, now.tm_mon, now.tm_mday)
    y = time.localtime(time.time() - 86400)
    yesterday = (y.tm_year, y.tm_mon, y.tm_mday)

    blocks = []
    for (yy, mm, dd), items in groups.items():
        if (yy, mm, dd) == today:
            label = "今天"
        elif (yy, mm, dd) == yesterday:
            label = "昨天"
        else:
            label = f"{mm}月{dd}日"
        gkey = f"{yy}-{mm:02d}-{dd:02d}"
        rows = []
        for s in items:
            t = time.localtime(s["updated_at"])
            ts = f"{t.tm_hour:02d}:{t.tm_min:02d}"
            act = " active" if s["chat_id"] == current_chat_id else ""
            safe_title = html.escape(s["title"], quote=True)
            safe_chat = html.escape(s["chat_id"], quote=True)
            rows.append(
                f'<div class="conv-item{act}" data-chat="{safe_chat}">'
                f'<span class="conv-time">{ts}</span>'
                f'<span class="conv-title">{safe_title}</span>'
                f'<span class="conv-del" data-chat="{safe_chat}">×</span>'
                f'</div>'
            )
        blocks.append(
            f'<div class="date-group" data-key="{gkey}">'
            f'<div class="date-title"><span class="date-toggle">▼</span>{label}</div>'
            f'<div class="date-list">{"".join(rows)}</div>'
            f'</div>'
        )
    return '<div class="sidebar-groups">' + "".join(blocks) + "</div>"



def _hide_q():
    return gr.update(value="", visible=False), gr.update(value="", visible=False), gr.update(value="", visible=False)


def bootstrap():
    """首次进入：固定用户 19836701050，恢复最近有内容的会话，返回初始 UI 状态。"""
    be = get_backend()
    store = be["store"]
    user = store.get_or_create_user(USER_ID, uid=f"u_{USER_ID}")
    sessions = store.list_sessions(user["id"])
    chat_id, hist = None, []
    for s in sessions:
        h = store.get_history(s["chat_id"])
        if h:
            chat_id, hist = s["chat_id"], h
            break
    if chat_id is None:
        chat_id = sessions[0]["chat_id"] if sessions else store.create_session(user["id"], "新对话")
        hist = [{"role": "assistant", "content": GREETING}]
    return hist, chat_id, user["id"], _radio_states(user["id"], chat_id), render_sidebar_html(user["id"], chat_id), *_hide_q()


def new_chat(user_id: str):
    be = get_backend()
    store = be["store"]
    chat_id = store.create_session(user_id, "新对话")
    gr.Info("已新建对话")
    return [{"role": "assistant", "content": GREETING}], chat_id, _radio_states(user_id, chat_id), render_sidebar_html(user_id, chat_id), "", *_hide_q()


def select_session(chat_id: str, user_id: str):
    if not chat_id:
        return gr.update(), chat_id, "", *_hide_q(), gr.update(), render_sidebar_html(user_id, chat_id)
    be = get_backend()
    store = be["store"]
    hist = store.get_history(chat_id)
    if not hist:
        hist = [{"role": "assistant", "content": GREETING}]
    
    return hist, chat_id, "", *_hide_q(), gr.update(), render_sidebar_html(user_id, chat_id)


def delete_session(chat_id: str, user_id: str):
    """删除指定会话；若删的是当前会话，自动切到最近会话。"""
    be = get_backend()
    store = be["store"]
    if chat_id:
        store.delete_session(chat_id)
    current = ""
    hist = [{"role": "assistant", "content": GREETING}]
    sessions = store.list_sessions(user_id)
    if sessions:
        current = sessions[0]["chat_id"]
        h = store.get_history(current)
        if h:
            hist = h
    return hist, current, *_hide_q(), _radio_states(user_id, current), render_sidebar_html(user_id, current)


def save_upload(temp_path):
    name = os.path.basename(temp_path)
    dest = os.path.join(UPLOAD_DIR, name)
    base, ext = os.path.splitext(name)
    i = 1
    while os.path.exists(dest):
        dest = os.path.join(UPLOAD_DIR, f"{base}_{i}{ext}")
        i += 1
    shutil.copy2(temp_path, dest)
    return os.path.basename(dest), dest


def render_tags(uploaded):
    if not uploaded:
        return '<div class="file-tags-empty">暂无已上传文件</div>'
    tags = []
    for idx, u in enumerate(uploaded):
        safe = html.escape(u["name"], quote=True)
        file_url = "/gradio_api/file=" + __import__("urllib.parse", fromlist=["quote"]).quote(u["path"], safe="")
        tags.append(
            f'<span class="file-tag">'
            f'<a class="file-tag-name" href="{file_url}" target="_blank" title="点击查看">{safe}</a>'
            f'<button class="file-tag-del" title="删除" onclick="removeFile(\'{idx}\')">×</button></span>'
        )
    return '<div class="file-tags">' + "".join(tags) + "</div>"


def remove_uploaded(idx, uploaded):
    try:
        idx = int(idx) if str(idx).isdigit() else -1
    except (ValueError, TypeError):
        idx = -1
    if 0 <= idx < len(uploaded):
        u = uploaded[idx]
        if os.path.dirname(os.path.abspath(u["path"])) == os.path.abspath(UPLOAD_DIR):
            try:
                os.remove(u["path"])
            except OSError:
                pass
        del uploaded[idx]
    return render_tags(uploaded), uploaded


def on_upload(files, chatbot, chat_id, user_id, uploaded):
    be = get_backend()
    store = be["store"]
    for f in files:
        saved_name, saved_path = save_upload(f)
        uploaded.append({"name": saved_name, "path": saved_path})
        
        result = be["process_upload"](saved_path, vector_store=be["vs"])
        chatbot.append({"role": "user", "content": f"[已上传文件] {saved_name}"})
        if result.get("ok"):
            msg = f"已接收并解析文件「{saved_name}」，切分 {result['chunks']} 个片段并入库，现在可以直接向我提问该文档内容。"
        else:
            msg = f"已保存文件「{saved_name}」，但解析失败：{result.get('error', '未知错误')}"
        chatbot.append({"role": "assistant", "content": msg})
        store.add_message(chat_id, "user", f"[已上传文件] {saved_name}")
        store.add_message(chat_id, "assistant", msg)
    return chatbot, render_tags(uploaded), uploaded


def respond(message, chatbot, chat_id, user_id, uploaded):
    be = get_backend()
    qa = be["qa"]
    router = be["router"]
    rec = be["recommender"]
    store = be["store"]

    user_msg = (message or "").strip()
    if not user_msg:
       
        return chatbot, "", *_hide_q(), gr.update(), gr.update()


    recent = store.history_for_router(chat_id)
    result = router.classify(user_msg, recent_context=recent)
    history = store.get_history(chat_id)

    if result.fallback_used:
        answer = f"抱歉，我对您的需求还不太确定。{result.fallback_info}"
    elif result.intent == "chitchat":
       
        if _is_factual_question(user_msg):
            answer, _sources = qa.ask(user_msg, history=history)
        else:
            answer = qa.chat_direct(user_msg, history)
    elif result.intent == "file_ops":
        if uploaded:
            names = "、".join(u["name"] for u in uploaded)
            answer = f"您当前已上传 {len(uploaded)} 个文件：{names}。您可以直接提问文档内容，或说明想对文件做什么操作。"
        else:
            answer = "您还没有上传文件。可以点击左下角「上传」按钮上传 PDF/DOCX/TXT 等文档，我会自动解析入库后为您解答。"
    else:
        
        answer, _sources = qa.ask(user_msg, history=history)

    chatbot.append({"role": "user", "content": user_msg})
    chatbot.append({"role": "assistant", "content": answer})
    store.add_message(chat_id, "user", user_msg)
    store.add_message(chat_id, "assistant", answer)
  
    title = user_msg[:10] + ("…" if len(user_msg) > 10 else "")
    store.touch_session(chat_id, title=title)

  
    questions = rec.recommend(user_msg, answer)
    qs = [gr.update(value=q, visible=True) for q in questions[:3]]
    while len(qs) < 3:
        qs.append(gr.update(value="", visible=False))
    
    return chatbot, "", *qs, _radio_states(user_id, chat_id), render_sidebar_html(user_id, chat_id)


def config_flag():
    import config
    return config.QUESTION_RECOMMEND


CSS = """
.gradio-container { max-width: 100% !important; font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif; }
body { background: #f0f2f5; }
#navbar { background: #1f2937 !important; border-radius: 0 !important; padding: 0 24px !important; height: 76px !important; align-items: center !important; display: flex !important; justify-content: space-between !important; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.navbar-user { color: #ffffff; font-size: 15px; font-weight: 500; white-space: nowrap; margin-right: auto; }
#exit-btn, #upload-btn, #send-btn { flex-grow: 0 !important; flex-shrink: 0 !important; flex-basis: auto !important; min-width: 0 !important; }
#exit-btn { background: #D65746 !important; color: #ffffff !important; border: none !important; border-radius: 18px !important; width: 96px !important; height: 43px !important; padding: 0 !important; font-size: 15px !important; font-weight: 500 !important; box-shadow: none !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; gap: 6px !important; }
#exit-btn:hover { background: #c04a3a !important; }
#exit-btn img { width: 18px !important; height: 18px !important; }
#upload-btn img { width: 20px !important; height: 20px !important; }
#send-btn img { width: 20px !important; height: 20px !important; }
#welcome-banner { text-align: center; background: linear-gradient(135deg, #eef4ff 0%, #ffffff 100%); padding: 16px 0; border-bottom: 1px solid #e5e7eb; }
#welcome-banner h1 { margin: 0; font-size: 24px; color: #1e3a8a; font-weight: 600; letter-spacing: 1px; }
#main-row { gap: 0 !important; }
#sidebar { background: #f7f8fa; border-right: 1px solid #e5e7eb; padding: 16px 12px !important; min-width: 220px !important; }
#new-chat-btn { background: #2563eb !important; color: #fff !important; border: none !important; border-radius: 8px !important; padding: 10px 0 !important; font-size: 15px !important; font-weight: 500 !important; box-shadow: 0 2px 6px rgba(37,99,235,0.3) !important; width: 100% !important; }
#new-chat-btn:hover { background: #1d4ed8 !important; }
.section-title { font-size: 13px; color: #6b7280; padding: 14px 4px 6px; font-weight: 500; }
#chat-col { background: #fff !important; padding: 0 !important; }
#chatbot { background: #fff !important; border: none !important; border-radius: 0 !important; }
#chatbot .message-row { padding: 10px 0 !important; }
#input-row { align-items: center !important; padding: 12px 16px !important; gap: 10px !important; border-top: 1px solid #e5e7eb; background: #fff; }
#file-area { padding: 10px 16px 0 !important; background: #fff; }
#file-tags { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; align-items: center !important; min-height: 30px; }
.file-tag { display: inline-flex !important; align-items: center !important; gap: 4px !important; background: #eef2ff !important; color: #1e40af !important; border: 1px solid #c7d2fe !important; border-radius: 14px !important; padding: 4px 10px !important; font-size: 13px !important; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-tag-del { background: transparent !important; border: none !important; color: #64748b !important; font-size: 15px !important; cursor: pointer !important; padding: 0 2px !important; line-height: 1 !important; }
.file-tag-del:hover { color: #dc2626 !important; }
.file-tag-name { color: #1e40af !important; text-decoration: none !important; cursor: pointer !important; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-tag-name:hover { text-decoration: underline !important; }
.file-tags-empty { color: #9ca3af !important; font-size: 13px !important; }
#upload-btn { background: #7CE0C4 !important; color: #ffffff !important; border: none !important; border-radius: 20px !important; width: 105px !important; height: 48px !important; padding: 0 !important; font-size: 16px !important; font-weight: 500 !important; box-shadow: none !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; gap: 6px !important; }
#upload-btn:hover { background: #6ad1b4 !important; }
#msg { border: 1px solid #d1d5db !important; border-radius: 6px !important; font-size: 14px !important; }
#send-btn { background: #5A90DC !important; color: #fff !important; border: none !important; border-radius: 20px !important; width: 104px !important; height: 48px !important; padding: 0 !important; font-size: 16px !important; font-weight: 500 !important; box-shadow: none !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; gap: 6px !important; }
#send-btn:hover { background: #4a7fca !important; }
#q-row { padding: 0 16px 10px !important; background: #fff; gap: 8px !important; }
#q-row button { background: #eef2ff !important; color: #1e40af !important; border: 1px solid #c7d2fe !important; border-radius: 14px !important; font-size: 13px !important; }
#q-row button:hover { background: #dbeafe !important; }

/* 聊天记录：按日期分组，每天一个框，框内竖向排列 */
#conv-radio { display: none !important; }
.sidebar-groups { display: flex; flex-direction: column; gap: 0; }
.date-group {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 6px 8px 8px !important;
    margin: 0 0 10px 0 !important;
}
.date-title {
    font-size: 13px !important;
    color: #6b7280 !important;
    font-weight: 600 !important;
    letter-spacing: 1px;
    padding: 4px 8px 6px !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}
.date-toggle {
    display: inline-block !important;
    width: 16px !important;
    height: 16px !important;
    line-height: 14px !important;
    text-align: center !important;
    font-size: 10px !important;
    color: #9ca3af !important;
    cursor: pointer !important;
    border-radius: 4px !important;
    user-select: none !important;
    transition: background 0.15s, color 0.15s;
}
.date-toggle:hover { background: #e5e7eb !important; color: #4f46e5 !important; }
.date-list { display: flex !important; flex-direction: column !important; gap: 2px !important; }
.date-list.collapsed { display: none !important; }
.conv-item {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 8px 10px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    position: relative !important;
    font-size: 14px !important;
    color: #374151 !important;
    transition: background 0.15s, color 0.15s;
}
.conv-item:hover { background: #eef2ff !important; color: #1e40af !important; }
.conv-item.active { background: #eef2ff !important; color: #1e40af !important; }
.conv-time { color: #9ca3af !important; font-size: 12px !important; white-space: nowrap !important; }
.conv-title { flex: 1 !important; overflow: hidden !important; text-overflow: ellipsis !important; white-space: nowrap !important; }
.conv-del {
    position: absolute !important;
    right: 6px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 18px !important;
    height: 18px !important;
    line-height: 16px !important;
    text-align: center !important;
    border-radius: 50% !important;
    color: #9ca3af !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    opacity: 0 !important;
    transition: opacity 0.15s, background 0.15s, color 0.15s;
}
.conv-item:hover .conv-del { opacity: 1 !important; }
.conv-del:hover { background: #fee2e2 !important; color: #dc2626 !important; }
"""

REMOVE_JS = """
window.removeFile = async function(name) {
  try {
    const sh = window.__gradio_session_hash__ || '';
    const resp = await fetch('/gradio_api/call/remove_uploaded', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({data: [name], session_hash: sh})
    });
    if (!resp.ok) { console.error('removeFile http', resp.status); return; }
    const ev = await resp.json();
    if (!ev.event_id) { console.error('removeFile no event_id'); return; }
    const sse = await fetch('/gradio_api/call/remove_uploaded/' + ev.event_id);
    if (!sse.ok) { console.error('removeFile sse http', sse.status); return; }
    const reader = sse.body.getReader();
    const decoder = new TextDecoder();
    let buf = '', output = null;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const m = buf.match(/event: complete\\s*\\ndata: ([^\\n]*)/s);
      if (m) { try { output = JSON.parse(m[1]); } catch (e) {} break; }
      if (buf.includes('event: error')) break;
    }
    if (output && output.length) { document.getElementById('file-tags').innerHTML = output[0]; }
  } catch (e) { console.error('removeFile error:', e); }
};

/* ===== 聊天记录：按日期分组——选择/删除/收起（纯事件委托，不与 Svelte 抢 DOM） ===== */
function triggerSelect(chatId) {
  var inputs = document.querySelectorAll('#conv-radio input[type="radio"]');
  var found = null;
  inputs.forEach(function(inp) {
    if (inp.value === chatId) { found = inp; } else { inp.checked = false; }
  });
  if (found) {
    found.checked = true;
    found.dispatchEvent(new Event('change', {bubbles: true}));
  }
}

function restoreCollapsed() {
  var st = {};
  try { st = JSON.parse(localStorage.getItem('date_collapsed') || '{}'); } catch (e) { st = {}; }
  document.querySelectorAll('.date-group').forEach(function(g) {
    var t = g.querySelector('.date-toggle');
    var list = g.querySelector('.date-list');
    if (!t || !list) return;
    var key = g.getAttribute('data-key');
    if (st[key]) { list.classList.add('collapsed'); t.textContent = '\\u25b6'; }
    else { list.classList.remove('collapsed'); t.textContent = '\\u25bc'; }
  });
}

document.addEventListener('click', function(e) {
  // 收起/展开
  var t = e.target.closest('.date-toggle');
  if (t) {
    e.preventDefault();
    var box = t.closest('.date-group');
    var list = box ? box.querySelector('.date-list') : null;
    if (!list) return;
    list.classList.toggle('collapsed');
    var collapsed = list.classList.contains('collapsed');
    t.textContent = collapsed ? '\\u25b6' : '\\u25bc';
    var st = {};
    try { st = JSON.parse(localStorage.getItem('date_collapsed') || '{}'); } catch (e2) { st = {}; }
    st[box.getAttribute('data-key')] = collapsed;
    localStorage.setItem('date_collapsed', JSON.stringify(st));
    return;
  }
  // 删除会话
  var del = e.target.closest('.conv-del');
  if (del) {
    e.preventDefault();
    e.stopPropagation();
    var chatId = del.getAttribute('data-chat');
    if (!chatId) return;
    if (!confirm('确认删除该对话？删除后不可恢复。')) return;
    var sh = window.__gradio_session_hash__ || '';
    fetch('/gradio_api/call/delete_session', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({data: [chatId], session_hash: sh})
    }).then(function(r) { return r.json(); }).then(function(ev) {
      if (ev && ev.event_id) {
        return fetch('/gradio_api/call/delete_session/' + ev.event_id);
      }
      throw new Error('no event_id');
    }).then(function(sse) { return sse.text(); }).catch(function(err) {
      console.error('delete_session error:', err);
    }).then(function() {
      window.location.reload();
    });
    return;
  }
  // 选择会话：通过隐藏 Radio 触发 Gradio change 事件
  var item = e.target.closest('.conv-item');
  if (item) {
    e.preventDefault();
    var chatId = item.getAttribute('data-chat');
    if (chatId) triggerSelect(chatId);
  }
});

// 页面加载完成后恢复收起状态（只执行一次）
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', restoreCollapsed);
} else {
  restoreCollapsed();
}
"""


with gr.Blocks(title="近屿智能课程咨询助手") as demo:

    chat_id = gr.State(value="")
    user_id = gr.State(value="")
    uploaded_files = gr.State(value=[])

    with gr.Row(elem_id="navbar"):
        gr.HTML(f'<div class="navbar-user">欢迎{USER_ID}！</div>')
        exit_btn = gr.Button("退出", elem_id="exit-btn", icon="assets/icon-exit.svg", size="sm")

    gr.HTML('<div id="welcome-banner"><h1>欢迎使用近屿智能课程咨询助手</h1></div>')

    with gr.Row(elem_id="main-row"):
        with gr.Column(elem_id="sidebar", scale=1):
            new_chat_btn = gr.Button("+ 新建对话", elem_id="new-chat-btn")
            gr.HTML('<div class="section-title">聊天记录</div>')
            sidebar_html = gr.HTML(render_sidebar_html("", None), elem_id="sidebar-groups")
            
            conv_radio = gr.Radio(choices=[], visible=True, label=None, show_label=False, elem_id="conv-radio")

        with gr.Column(elem_id="chat-col", scale=4):
            chatbot = gr.Chatbot(value=[{"role": "assistant", "content": GREETING}], elem_id="chatbot", height=520, show_label=False, avatar_images=(None, None))

            with gr.Row(elem_id="file-area"):
                file_tags = gr.HTML(render_tags([]), elem_id="file-tags")

            with gr.Row(elem_id="input-row"):
                upload_btn = gr.UploadButton("上传", file_types=[".pdf", ".docx", ".txt", ".pptx", ".html", ".ipynb"], file_count="multiple", elem_id="upload-btn", icon="assets/icon-upload.svg")
                msg = gr.Textbox(placeholder="上传的文档不超过32M，格式：pdf/docx/txt/pptx/html/ipynb。输入你的消息...", show_label=False, elem_id="msg", scale=6, container=False)
                send_btn = gr.Button("发送", elem_id="send-btn", scale=1, icon="assets/icon-send.svg")

            with gr.Row(elem_id="q-row"):
                qbtn1 = gr.Button("", visible=False, size="sm")
                qbtn2 = gr.Button("", visible=False, size="sm")
                qbtn3 = gr.Button("", visible=False, size="sm")

  
    demo.load(fn=bootstrap, inputs=None, outputs=[chatbot, chat_id, user_id, conv_radio, sidebar_html, qbtn1, qbtn2, qbtn3])

    new_chat_btn.click(fn=new_chat, inputs=[user_id], outputs=[chatbot, chat_id, conv_radio, sidebar_html, msg, qbtn1, qbtn2, qbtn3], api_name="new_chat")
    conv_radio.change(fn=select_session, inputs=[conv_radio, user_id], outputs=[chatbot, chat_id, msg, qbtn1, qbtn2, qbtn3, conv_radio, sidebar_html], api_name="select_session")

    exit_btn.click(fn=lambda: gr.Info("已退出（演示）"), inputs=None, outputs=None)

    upload_btn.upload(fn=on_upload, inputs=[upload_btn, chatbot, chat_id, user_id, uploaded_files], outputs=[chatbot, file_tags, uploaded_files], api_name="upload_files")

    remove_name = gr.Textbox("", visible=False)
    remove_trigger = gr.Button("remove", visible=False)
    remove_trigger.click(fn=remove_uploaded, inputs=[remove_name, uploaded_files], outputs=[file_tags, uploaded_files], api_name="remove_uploaded")

    del_chat_id = gr.Textbox("", visible=False)
    delete_trigger = gr.Button("delete", visible=False)
    delete_trigger.click(fn=delete_session, inputs=[del_chat_id, user_id], outputs=[chatbot, chat_id, qbtn1, qbtn2, qbtn3, conv_radio, sidebar_html], api_name="delete_session")

    send_btn.click(fn=respond, inputs=[msg, chatbot, chat_id, user_id, uploaded_files], outputs=[chatbot, msg, qbtn1, qbtn2, qbtn3, conv_radio, sidebar_html])
    msg.submit(fn=respond, inputs=[msg, chatbot, chat_id, user_id, uploaded_files], outputs=[chatbot, msg, qbtn1, qbtn2, qbtn3, conv_radio, sidebar_html])

    qbtn1.click(fn=lambda v: v, inputs=[qbtn1], outputs=[msg])
    qbtn2.click(fn=lambda v: v, inputs=[qbtn2], outputs=[msg])
    qbtn3.click(fn=lambda v: v, inputs=[qbtn3], outputs=[msg])


if __name__ == "__main__":
   
    print("正在初始化后端（加载 embedding 模型，约 7 秒）…", flush=True)
    get_backend()
    print("后端就绪，正在启动界面…", flush=True)
    demo.launch(
        css=CSS,
        theme=gr.themes.Base(),
        inbrowser=True,
        js=REMOVE_JS,
        allowed_paths=[UPLOAD_DIR],
        footer_links=[],
    )
