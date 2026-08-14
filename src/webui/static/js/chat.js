/* chat.js — 聊天框：POST 提交 + fetch 流式读取 SSE */
(function (window) {
  'use strict';

  const messagesEl = document.getElementById('chat-messages');
  const inputEl = document.getElementById('chat-input');
  const sendBtn = document.getElementById('btn-send');

  function appendMsg(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  // 流式读取 SSE：POST 不能用 EventSource，用 fetch + ReadableStream
  async function streamChat(message) {
    appendMsg('user', message);
    const aiMsg = appendMsg('assistant', '');
    aiMsg.classList.add('streaming');

    const res = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'chat', message }),
    });
    const { job_id } = await res.json();
    if (!job_id) {
      aiMsg.textContent = '❌ 无法创建聊天任务';
      aiMsg.classList.remove('streaming');
      return;
    }

    const streamRes = await fetch(`/api/jobs/${job_id}/stream`);
    if (!streamRes.body) {
      aiMsg.textContent = '❌ 流不可用';
      aiMsg.classList.remove('streaming');
      return;
    }

    const reader = streamRes.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buf = '';
    let reply = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // 解析 SSE 事件（event: X\ndata: {...}\n\n）
        let idx;
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const rawEvent = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const dataMatch = rawEvent.match(/^data:\s*(.+)$/m);
          if (!dataMatch) continue;
          const data = JSON.parse(dataMatch[1]);
          if (data.line !== undefined) {
            reply += data.line + '\n';
            aiMsg.textContent = reply;
          } else if (data.result !== undefined) {
            reply = data.result.reply || '';
            aiMsg.textContent = reply;
          } else if (data.error) {
            reply += '\n❌ ' + data.error;
            aiMsg.textContent = reply;
          }
        }
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    } finally {
      aiMsg.classList.remove('streaming');
      if (!aiMsg.textContent.trim()) aiMsg.textContent = '（无响应）';
    }
  }

  function send() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    sendBtn.disabled = true;
    streamChat(text)
      .catch(e => appendMsg('sys', `⚠️ 发送失败: ${e.message}`))
      .finally(() => { sendBtn.disabled = false; inputEl.focus(); });
  }

  sendBtn.addEventListener('click', send);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });

  // 欢迎消息
  appendMsg('sys', '💬 输入自然语言指令与系统交互，如"分析茅台"、"对比茅台和五粮液"、"扫描热门"、"训练600519"');

  window.Chat = { appendMsg };
})(window);
