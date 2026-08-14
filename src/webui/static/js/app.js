/* app.js — 主控逻辑：选股流程 / 侧边栏双 tab / RL 训练 / 搜索联想 */
(function () {
  'use strict';

  const AppState = {
    currentCode: null,
    listMode: 0,            // 0=历史查询 1=自选股
    activeJobs: {},         // job_id -> {type, code}
  };
  window.AppState = AppState;

  const LIST_MODES = ['history', 'watchlist'];
  const LIST_NAMES = ['历史查询', '自选股'];

  // ═══ 搜索联想 ═══
  const searchInput = document.getElementById('stock-search');
  const suggestEl = document.getElementById('search-suggest');
  let searchTimer = null;

  searchInput.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const q = searchInput.value.trim();
    if (!q) { suggestEl.classList.add('hidden'); return; }
    searchTimer = setTimeout(async () => {
      try {
        const { results } = await API.search(q, 8);
        if (!results.length) { suggestEl.classList.add('hidden'); return; }
        suggestEl.innerHTML = results.map(r =>
          `<div class="item" data-code="${r.code}"><span>${r.name}</span><span class="code">${r.code}</span></div>`
        ).join('');
        suggestEl.classList.remove('hidden');
        suggestEl.querySelectorAll('.item').forEach(el => {
          el.addEventListener('click', () => {
            openStock(el.dataset.code);
            searchInput.value = '';
            suggestEl.classList.add('hidden');
          });
        });
      } catch (e) { suggestEl.classList.add('hidden'); }
    }, 300);
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-box')) suggestEl.classList.add('hidden');
  });

  // ═══ 侧边栏双 tab 切换 ═══
  function switchList(mode) {
    AppState.listMode = mode;
    document.getElementById('list-name').textContent = LIST_NAMES[mode];
    loadSidebar();
  }
  document.getElementById('prev-list').addEventListener('click', () => {
    switchList((AppState.listMode - 1 + LIST_MODES.length) % LIST_MODES.length);
  });
  document.getElementById('next-list').addEventListener('click', () => {
    switchList((AppState.listMode + 1) % LIST_MODES.length);
  });

  async function loadSidebar() {
    const box = document.getElementById('stock-list');
    const mode = LIST_MODES[AppState.listMode];
    try {
      let items = [];
      if (mode === 'history') {
        const { items: h } = await API.history(50);
        items = h || [];
      } else {
        const { items: w } = await API.watchlist();
        items = w || [];
      }
      renderSidebar(items, box);
    } catch (e) {
      box.innerHTML = `<div class="empty-hint">加载失败: ${e.message}</div>`;
    }
  }

  function renderSidebar(items, box) {
    if (!items.length) {
      box.innerHTML = `<div class="empty-hint">${AppState.listMode === 0 ? '暂无历史记录，搜索股票后自动加入' : '暂无自选股，搜索后点击加入'}</div>`;
      return;
    }
    box.innerHTML = items.map(it => `
      <div class="stock-item ${it.code === AppState.currentCode ? 'active' : ''}" data-code="${it.code}">
        <span class="s-name">${it.name || it.code}</span>
        <span>
          <span class="s-code">${it.code}</span>
          ${it.is_trained ? '<span class="s-trained" title="已训练 RL 模型">✓</span>' : ''}
          ${it.count ? `<span class="s-count">${it.count}次</span>` : ''}
        </span>
      </div>
    `).join('');

    box.querySelectorAll('.stock-item').forEach(el => {
      el.addEventListener('click', () => openStock(el.dataset.code));
    });
  }

  // ═══ 路由（hash 深链） ═══
  function parseCode(code) {
    // (market, symbol) —— 与后端 market_catalog.parse_code 等价
    if (code.startsWith('hk:')) return ['hk', code.slice(3)];
    if (code.startsWith('us:')) return ['us', code.slice(3)];
    return ['a', code];
  }
  function parseHash() {
    const h = location.hash.replace('#', '').trim();
    if (!h || h === '/' || h === '') return { view: 'home' };
    const m = h.match(/^\/stock\/(.+)$/);
    if (m) return { view: 'stock', code: decodeURIComponent(m[1]) };
    if (/^\d{6}$/.test(h)) return { view: 'stock', code: h };  // 旧深链兼容
    return { view: 'home' };
  }
  function navigate(code) {
    const target = code ? '/stock/' + encodeURIComponent(code) : '/';
    if (location.hash !== '#' + target) {
      location.hash = target;
    }
  }

  function showHome() {
    document.getElementById('home-view').classList.remove('hidden');
    document.getElementById('detail-view').classList.add('hidden');
  }

  // ═══ 选股主流程 ═══
  async function openStock(code) {
    if (!code) return;
    AppState.currentCode = code;
    navigate(code);
    highlightSidebar(code);

    // 视图切换 + 市场标签
    document.getElementById('home-view').classList.add('hidden');
    document.getElementById('detail-view').classList.remove('hidden');
    const [market] = parseCode(code);
    const mtag = document.getElementById('stock-market-tag');
    mtag.textContent = { a: 'A股', hk: '港股', us: '美股' }[market] || 'A股';
    mtag.className = 'chip ' + (market === 'a' ? '' : '');
    // 港股/美股隐藏 RL 相关按钮与自选按钮（自选仅支持 A 股 6 位代码）
    const nonA = market !== 'a';
    document.getElementById('btn-train').style.display = nonA ? 'none' : '';
    document.getElementById('btn-refresh-rl').style.display = nonA ? 'none' : '';
    document.getElementById('btn-predict').style.display = nonA ? 'none' : '';
    watchBtn.style.display = nonA ? 'none' : '';

    document.getElementById('stock-code').textContent = code;
    // 同步自选按钮状态
    if (!nonA) updateWatchBtn(code);
    Kline.showLoading();
    Cards.setCardLoading('rules', '加载中…');
    Cards.setCardLoading('skills', '加载中…');
    Cards.setCardLoading('rl', '加载中…');
    Cards.setCardLoading('llm', '点击卡片右上角"分析"按钮触发');
    Cards.setCardLoading('fused', '点击卡片右上角"预测"按钮触发');

    // 快速数据：K线 + 实时行情（并行，不阻塞）
    const [kline, quote] = await Promise.all([
      API.kline(code).catch(e => ({ error: e.message })),
      API.quote(code).catch(e => ({ error: e.message })),
    ]);
    if (AppState.currentCode !== code) return;  // 期间已切换股票

    if (kline.error) {
      document.getElementById('stock-name').textContent = code;
    } else {
      document.getElementById('stock-name').textContent = kline.name || code;
      document.getElementById('stock-code').textContent = code;
      Kline.render(kline);
    }
    if (quote.error) {
      document.getElementById('quote-cards').innerHTML = `<div class="empty-hint">行情获取失败: ${quote.error}</div>`;
    } else {
      Cards.renderQuote(quote);
    }

    // 慢速数据：分析 / 技能 / RL 各自独立异步加载（哪个先完成先显示）
    // 竞态保护：回调时若已切换股票则丢弃旧结果
    const stillCurrent = () => AppState.currentCode === code;
    API.analyze(code)
      .then(r => { if (stillCurrent()) Cards.renderCard('rules', r); })
      .catch(e => { if (stillCurrent()) Cards.renderCard('rules', { error: e.message }); });
    API.skills(code)
      .then(r => { if (stillCurrent()) Cards.renderCard('skills', r); })
      .catch(e => { if (stillCurrent()) Cards.renderCard('skills', { error: e.message }); });
    API.rl(code)
      .then(r => { if (stillCurrent()) Cards.renderCard('rl', r); })
      .catch(e => { if (stillCurrent()) Cards.renderCard('rl', { error: e.message }); });

    // 刷新侧边栏（记录本次查询）
    if (AppState.listMode === 0) loadSidebar();
    checkActiveJobsFor(code);
  }

  function highlightSidebar(code) {
    document.querySelectorAll('.stock-item').forEach(el => {
      el.classList.toggle('active', el.dataset.code === code);
    });
  }

  // ═══ RL 刷新按钮（强制绕过缓存重新获取） ═══
  document.getElementById('btn-refresh-rl').addEventListener('click', async () => {
    const code = AppState.currentCode;
    if (!code) return;
    Cards.setCardLoading('rl', '刷新中…');
    try {
      const r = await API.rl(code, true);
      Cards.renderCard('rl', r);
    } catch (e) {
      Cards.renderCard('rl', { error: e.message });
    }
  });

  // RL 刷新（供训练完成等场景调用，强制绕过缓存）
  function refreshRlCard(code) {
    API.rl(code, true).then(r => Cards.renderCard('rl', r)).catch(() => {});
  }

  // 规则引擎 / 技能分析 编辑按钮 → 打开参数编辑弹窗
  document.getElementById('btn-edit-rules').addEventListener('click', () => {
    openEditModal('rules');
  });
  document.getElementById('btn-edit-skills').addEventListener('click', () => {
    openEditModal('skills');
  });

  // ═══ 规则编辑弹窗 ═══
  const modal = document.getElementById('edit-modal');
  const modalTitle = document.getElementById('edit-modal-title');
  const modalBody = document.getElementById('edit-modal-body');
  let modalMode = 'rules';
  let modalDirty = false;

  const RULES_FIELDS = [
    { key: 'trend_ma_short', label: '短期均线周期', desc: '趋势跟踪 MA 短周期', step: 1 },
    { key: 'trend_ma_medium', label: '中期均线周期', desc: '趋势跟踪 MA 中周期', step: 1 },
    { key: 'trend_ma_long', label: '长期均线周期', desc: '趋势跟踪 MA 长周期', step: 1 },
    { key: 'rsi_overbought', label: 'RSI 超买阈值', desc: '超过则视为超买(卖点)', step: 1 },
    { key: 'rsi_oversold', label: 'RSI 超卖阈值', desc: '低于则视为超卖(买点)', step: 1 },
    { key: 'volume_surge_ratio', label: '放量倍数', desc: '量比超过视为放量', step: 0.1 },
    { key: 'fusion_llm_skills', label: 'LLM+技能权重', desc: '三路融合 LLM 权重', step: 0.05 },
    { key: 'fusion_rl', label: 'RL 权重', desc: '三路融合 RL 权重', step: 0.05 },
    { key: 'fusion_rule', label: '规则引擎权重', desc: '三路融合规则权重', step: 0.05 },
    { key: 'kelly_odds', label: '凯利赔率 b', desc: '止盈20%/止损5% → b=4', step: 0.5 },
    { key: 'max_position', label: '单标的最大仓位', desc: '上限比例 0.20', step: 0.01 },
  ];
  const SKILLS_FIELDS = [
    { key: 'enabled', label: '技能引擎开关', desc: 'true/false', type: 'check' },
    { key: 'confidence_threshold', label: '信号置信度阈值', desc: '低于则忽略信号', step: 0.05 },
  ];

  function openEditModal(mode) {
    modalMode = mode;
    modalDirty = false;
    modalTitle.textContent = mode === 'rules' ? '✏️ 规则引擎参数' : '✏️ 技能分析参数';
    const fields = mode === 'rules' ? RULES_FIELDS : SKILLS_FIELDS;

    const load = mode === 'rules' ? API.get('/config/rules') : API.get('/config/skills');
    load.then(cfg => {
      modalBody.innerHTML = fields.map(f => {
        let val = cfg[f.key];
        if (f.type === 'check') {
          const checked = !!val;
          return `<div class="edit-field"><label>${f.label}<div class="field-desc">${f.desc || ''}</div></label>
            <select data-key="${f.key}" class="edit-select">
              <option value="true" ${checked ? 'selected' : ''}>开启</option>
              <option value="false" ${!checked ? 'selected' : ''}>关闭</option>
            </select></div>`;
        }
        const num = (val == null ? '' : val);
        return `<div class="edit-field"><label>${f.label}<div class="field-desc">${f.desc || ''}</div></label>
          <input type="number" data-key="${f.key}" value="${num}" step="${f.step || 'any'}" min="0"></div>`;
      }).join('');
      modal.classList.remove('hidden');
    }).catch(e => {
      window.Chat && window.Chat.appendMsg('sys', '⚠️ 参数加载失败: ' + e.message);
    });
  }

  document.getElementById('btn-modal-close').addEventListener('click', () => {
    modal.classList.add('hidden');
  });
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.add('hidden');
  });

  document.getElementById('btn-modal-save').addEventListener('click', async () => {
    const params = {};
    modalBody.querySelectorAll('[data-key]').forEach(el => {
      const key = el.dataset.key;
      if (el.tagName === 'SELECT') params[key] = el.value === 'true';
      else params[key] = parseFloat(el.value);
    });
    try {
      if (modalMode === 'rules') {
        await API.put('/config/rules', { params });
        window.Chat && window.Chat.appendMsg('sys', '✅ 规则引擎参数已保存（下次分析生效）');
      } else {
        await API.put('/config/skills', { params });
        window.Chat && window.Chat.appendMsg('sys', '✅ 技能分析参数已保存');
      }
      modal.classList.add('hidden');
      // 清前端缓存，确保重新拉取新参数下的分析结果
      API.clearCache();
      refreshAllCards();
    } catch (e) {
      window.Chat && window.Chat.appendMsg('sys', '❌ 保存失败: ' + e.message);
    }
  });

  document.getElementById('btn-modal-reset').addEventListener('click', async () => {
    if (modalMode === 'rules') {
      await API.post('/config/rules/reset');
      window.Chat && window.Chat.appendMsg('sys', '↩️ 规则参数已恢复默认');
      modal.classList.add('hidden');
      API.clearCache();
      refreshAllCards();
    }
  });

  function refreshAllCards() {
    const code = AppState.currentCode;
    if (!code) return;
    Cards.setCardLoading('rules', '参数已更新，重新分析中…');
    Cards.setCardLoading('skills', '参数已更新，重新分析中…');
    API.analyze(code).then(r => Cards.renderCard('rules', r)).catch(() => {});
    API.skills(code).then(r => Cards.renderCard('skills', r)).catch(() => {});
  }

  // ═══ 加自选 / 移除自选 ═══
  const watchBtn = document.getElementById('btn-watch');

  async function updateWatchBtn(code) {
    try {
      const { items } = await API.watchlist();
      const inWatch = (items || []).some(it => it.code === code);
      watchBtn.textContent = inWatch ? '★ 已自选' : '☆ 加自选';
      watchBtn.title = inWatch ? '从自选移除' : '加入自选';
      watchBtn.dataset.inWatch = inWatch ? '1' : '0';
    } catch (e) { /* ignore */ }
  }

  watchBtn.addEventListener('click', async () => {
    const code = AppState.currentCode;
    if (!code) return;
    const adding = watchBtn.dataset.inWatch !== '1';
    watchBtn.disabled = true;
    try {
      if (adding) {
        await API.addWatch(code);
        window.Chat && window.Chat.appendMsg('sys', `⭐ 已将 ${code} 加入自选`);
      } else {
        await API.removeWatch(code);
        window.Chat && window.Chat.appendMsg('sys', `已从自选移除 ${code}`);
      }
      await updateWatchBtn(code);
      if (AppState.listMode === 1) loadSidebar();  // 正在查看自选 tab 时刷新
    } catch (e) {
      window.Chat && window.Chat.appendMsg('sys', '❌ ' + (adding ? '加入自选失败' : '移除自选失败') + ': ' + e.message);
    } finally {
      watchBtn.disabled = false;
    }
  });

  // ═══ 顶栏按钮：LLM / 预测 ═══
  document.getElementById('btn-llm').addEventListener('click', async () => {
    const code = AppState.currentCode;
    if (!code) return;
    Cards.setCardLoading('llm', 'LLM 深度分析中（可能较慢）…');
    try {
      const r = await API.llm(code);
      Cards.renderCard('llm', r);
    } catch (e) {
      Cards.renderCard('llm', { error: e.message });
    }
  });

  document.getElementById('btn-predict').addEventListener('click', async () => {
    const code = AppState.currentCode;
    if (!code) return;
    Cards.setCardLoading('fused', '三路融合分析中…');
    try {
      const { job_id } = await API.submitJob('predict', { codes: [code], use_llm: true });
      await pollJob(job_id, (job) => {
        if (job.status === 'done' && job.result) {
          Cards.renderCard('fused', job.result);
          return true;
        }
        if (job.status === 'error') {
          Cards.renderCard('fused', { error: job.error });
          return true;
        }
        return false;
      });
    } catch (e) {
      Cards.renderCard('fused', { error: e.message });
    }
  });

  // ═══ RL 训练 ═══
  const trainBtn = document.getElementById('btn-train');
  const trainProgress = document.getElementById('train-progress');

  trainBtn.addEventListener('click', async () => {
    const code = AppState.currentCode;
    if (!code) return;
    trainBtn.disabled = true;
    trainProgress.classList.remove('hidden');
    const fill = document.getElementById('train-progress-fill');
    const text = document.getElementById('train-progress-text');

    try {
      const res = await API.submitJob('train', { code, timesteps: 50000 });
      if (res.error && res.code === 'conflict') {
        text.textContent = `该股票已在训练中 (job ${res.job_id})`;
        // 订阅已存在 job
        pollJob(res.job_id, job => handleTrainProgress(job, fill, text, code));
      } else if (res.job_id) {
        AppState.activeJobs[res.job_id] = { type: 'train', code };
        pollJob(res.job_id, job => handleTrainProgress(job, fill, text, code));
      } else {
        text.textContent = '❌ ' + (res.error || '训练提交失败');
      }
    } catch (e) {
      text.textContent = '❌ ' + e.message;
    }
  });

  document.getElementById('btn-cancel-train').addEventListener('click', async () => {
    // 取消当前 code 的训练 job
    for (const [id, job] of Object.entries(AppState.activeJobs)) {
      if (job.type === 'train' && job.code === AppState.currentCode) {
        try { await API.cancelJob(id); } catch (e) { /* ignore */ }
      }
    }
  });

  function handleTrainProgress(job, fill, text, code) {
    if (job.status === 'running' || job.status === 'queued') {
      fill.style.width = Math.round((job.progress || 0) * 100) + '%';
      text.textContent = job.message || '训练中…';
      return false;
    }
    if (job.status === 'done') {
      fill.style.width = '100%';
      text.textContent = job.cancelled
        ? '训练已取消（模型未完整训练）'
        : '✅ 训练完成，刷新 RL 分析…';
      // 强制刷新 RL 卡片（绕过后端缓存，训练结果才可见）
      if (!job.cancelled) refreshRlCard(code);
      if (AppState.listMode === 0) loadSidebar();
      setTimeout(() => trainProgress.classList.add('hidden'), 4000);
      trainBtn.disabled = false;
      return true;
    }
    if (job.status === 'error') {
      text.textContent = '❌ ' + (job.error || '训练失败');
      trainBtn.disabled = false;
      return true;
    }
    if (job.status === 'cancelled') {
      text.textContent = '训练已取消';
      trainBtn.disabled = false;
      return true;
    }
    return false;
  }

  // ═══ 通用 Job 轮询 ═══
  async function pollJob(jobId, onUpdate, interval = 1500) {
    for (let i = 0; i < 600; i++) {  // 最多 15 分钟
      await new Promise(r => setTimeout(r, interval));
      let job;
      try {
        job = await API.jobStatus(jobId);
      } catch (e) { continue; }
      if (job.error) { onUpdate({ status: 'error', error: job.error }); return; }
      if (onUpdate(job)) {
        delete AppState.activeJobs[jobId];
        return;
      }
    }
  }

  function checkActiveJobsFor(code) {
    // 恢复训练中状态（页面刷新后）
    Object.entries(AppState.activeJobs).forEach(([id, job]) => {
      if (job.code === code) {
        trainBtn.disabled = true;
        trainProgress.classList.remove('hidden');
      }
    });
  }

  // ═══ 大盘 / LLM 状态 chip ═══
  async function loadChips() {
    try {
      const m = await API.marketState();
      const chip = document.getElementById('market-chip');
      const st = m.market_state || 'unknown';
      chip.textContent = `大盘: ${st}${m.ret_20d ? ` (${m.ret_20d >= 0 ? '+' : ''}${m.ret_20d.toFixed(1)}%)` : ''}`;
      chip.className = 'chip ' + (st === 'bull' ? 'bull' : st === 'bear' ? 'bear' : st === 'range' ? 'range' : '');
    } catch (e) { /* ignore */ }
    try {
      const l = await API.llmStatus();
      const state = !l.enabled ? '⚪ 关闭' : (l.has_api_key ? '🟢' : '🔴');
      document.getElementById('llm-chip').textContent = `LLM: ${state}`;
      document.getElementById('llm-chip').title = l.enabled
        ? `模型: ${l.model}\nAPI: ${l.api_base}`
        : 'LLM 已关闭（点击 ⚙️ LLM 配置可开启）';
    } catch (e) { /* ignore */ }
  }

  // ═══ LLM 配置弹窗 ═══
  const llmModal = document.getElementById('llm-modal');
  const llmTestResult = document.getElementById('llm-test-result');

  function collectLlmConfig() {
    return {
      api_base: document.getElementById('llm-api-base').value.trim(),
      model: document.getElementById('llm-model').value.trim(),
      api_key: document.getElementById('llm-api-key').value.trim(),
      enabled: document.getElementById('llm-enabled').value === 'true',
    };
  }

  document.getElementById('btn-llm-config').addEventListener('click', async () => {
    llmTestResult.classList.add('hidden');
    llmTestResult.textContent = '';
    try {
      const cfg = await API.get('/llm/config');
      document.getElementById('llm-api-base').value = cfg.api_base || '';
      document.getElementById('llm-model').value = cfg.model || '';
      document.getElementById('llm-api-key').value = cfg.api_key || '';
      document.getElementById('llm-enabled').value = cfg.enabled ? 'true' : 'false';
      llmModal.classList.remove('hidden');
    } catch (e) {
      window.Chat && window.Chat.appendMsg('sys', '❌ 加载 LLM 配置失败: ' + e.message);
    }
  });
  document.getElementById('btn-llm-close').addEventListener('click', () => {
    llmModal.classList.add('hidden');
  });
  llmModal.addEventListener('click', (e) => {
    if (e.target === llmModal) llmModal.classList.add('hidden');
  });

  document.getElementById('btn-llm-test').addEventListener('click', async () => {
    llmTestResult.classList.remove('hidden');
    llmTestResult.style.color = '';
    llmTestResult.textContent = '⏳ 正在连接…';
    try {
      const r = await API.post('/llm/test', { config: collectLlmConfig() });
      llmTestResult.textContent = r.ok ? '✅ 连接成功: ' + (r.reply || r.model) : '❌ ' + (r.error || '连接失败');
      llmTestResult.style.color = r.ok ? '#3fb950' : '#f85149';
    } catch (e) {
      llmTestResult.textContent = '❌ ' + e.message;
      llmTestResult.style.color = '#f85149';
    }
  });

  document.getElementById('btn-llm-save').addEventListener('click', async () => {
    try {
      await API.put('/llm/config', { config: collectLlmConfig() });
      window.Chat && window.Chat.appendMsg('sys', '✅ LLM 配置已保存');
      llmModal.classList.add('hidden');
      loadChips();  // 刷新 LLM 状态 chip
    } catch (e) {
      window.Chat && window.Chat.appendMsg('sys', '❌ 保存 LLM 配置失败: ' + e.message);
    }
  });

  // ═══ 模块说明弹窗（卡片标题 ℹ️ 按钮）═══
  const helpModal = document.getElementById('help-modal');
  const helpModalTitle = document.getElementById('help-modal-title');
  const helpModalBody = document.getElementById('help-modal-body');

  const HELP_CONTENT = {
    rules: {
      title: '📐 规则引擎',
      html: `<p>硬规则引擎用 <b>10 条投票规则 + 1 条否决规则</b> 判断买卖方向，分 5 大类别：</p>
<ul>
  <li><b>趋势跟踪</b>：均线排列 (MA5/20/60)、MACD 金叉死叉、ADX 趋势强度</li>
  <li><b>震荡指标</b>：KDJ 超买超卖、RSI 超买超卖（阈值随大盘自适应）</li>
  <li><b>通道波动</b>：布林带 %B 位置、带宽收窄/扩张</li>
  <li><b>量价分析</b>：量价配合、天量/地量/量价背离等异常</li>
  <li><b>价格结构</b>：支撑阻力位、涨跌停与跳空缺口</li>
</ul>
<p><b>否决规则</b>：止损触发（亏损 ≥5%）直接覆盖投票结果，风控优先。</p>
<p><b>市场自适应</b>：牛市增强趋势权重、熊市下移超卖阈值，动态抑制假信号。</p>
<p class="help-note">类内取最强信号，类间加权投票 → 输出方向 + 强度。</p>`,
    },
    skills: {
      title: '📊 技能分析',
      html: `<p><b>11 个独立决策技能</b> 各自扫描数据，输出 {信号, 置信度, 说明}：</p>
<ul>
  <li><b>形态识别</b>：锤子线、吞没、十字星、启明/黄昏星、长影线等 K 线组合</li>
  <li><b>背离检测</b>：RSI 背离、MACD 背离 + 金叉死叉</li>
  <li><b>趋势分析</b>：均线+布林+前高突破、支撑阻力位</li>
  <li><b>动量分析</b>：多周期动量、均线排列、动量衰竭</li>
  <li><b>量价分析</b>：放量/缩量与涨跌的 7 种组合场景</li>
  <li><b>波动率</b>：ATR 趋势、布林收窄/扩张、均值回归 Z-score</li>
  <li><b>资金流向</b>：主力净流入/流出、连续流入、价资背离</li>
  <li><b>波动率区间</b>：低波压缩/正常/高波恐慌三档可交易性判断</li>
  <li><b>多周期共识</b>：日/周/月线趋势共振，防假信号</li>
  <li><b>新闻舆情</b>：个股新闻情绪分析（LLM 或本地关键词）</li>
</ul>
<p>结果汇总后，通过 <b>5 维本地评分卡</b> 或 <b>LLM 全局推理</b> 融合成综合信号。</p>`,
    },
    llm: {
      title: '🧠 LLM 分析',
      html: `<p><b>LLM 深度分析</b>：把 11 个技能的输出 + 技术指标快照 + 近期新闻交给大模型，让它理解信号间的<b>矛盾与印证</b>，输出独立的买卖决策。</p>
<ul>
  <li>独立判断：不是简单加权，而是从全局理解多信号是否一致</li>
  <li>消息面：结合近期新闻舆情判断对技术面的影响</li>
  <li>价格预测：给出短期/中期/中长期/长期四档方向与幅度</li>
</ul>
<p>点击卡片右上角 <b>🧠 分析</b> 触发（较慢）。</p>
<p class="help-note">⚠️ 在 ⚙️ LLM 配置关闭后，此卡自动走本地增强引擎（11 技能 5 维评分卡），不再调用 LLM。</p>`,
    },
    rl: {
      title: '🎮 RL 分析',
      html: `<p><b>PPO 强化学习智能体</b> 从历史数据自主学习买卖策略：</p>
<ul>
  <li><b>算法</b>：PPO（stable-baselines3）</li>
  <li><b>环境</b>：单股环境，动作 = {持有 0 / 买入 1 / 卖出 2}</li>
  <li><b>观察</b>：60 天窗口，15 维特征（OHLCV + MA + MACD + RSI + 布林等）</li>
  <li><b>奖励</b>：持仓盈亏 + 交易成本惩罚 + 持仓时间约束</li>
  <li><b>推理</b>：跑完整 episode → 统计买/卖/持占比 → 多数方向为最终动作</li>
</ul>
<p><b>模型生命周期</b>：&lt;15 天增量微调、15-60 天全量重训、&gt;60 天自动过期重建。</p>
<p><b>未训练时</b>：返回"未训练"，三路融合中此路自动降级。</p>`,
    },
    fused: {
      title: '🔗 综合建议',
      html: `<p><b>三路融合</b>：LLM+Skills / 硬规则 / RL 三方独立决策 → 加权融合出<b>最终买卖建议</b>。</p>
<ul>
  <li><b>权重</b>：LLM+Skills 40% + 硬规则 30% + RL 30%（可在规则引擎 ✏️ 编辑里调整）</li>
  <li><b>共识度</b>：三路一致 = 高（可信）、两路一致 = 中（可参考）、三路分歧 = 低（建议谨慎）</li>
  <li><b>仓位</b>：凯利公式 + 大盘状态限仓，输出建议仓位比例</li>
</ul>
<p>点击卡片右上角 <b>🔗 预测</b> 触发；RL 未训练时该路自动降级为 hold。</p>`,
    },
  };

  function openHelp(helpKey) {
    const h = HELP_CONTENT[helpKey];
    if (!h) return;
    helpModalTitle.textContent = h.title;
    helpModalBody.innerHTML = h.html;
    helpModal.classList.remove('hidden');
  }
  document.querySelectorAll('.btn-info').forEach(btn => {
    btn.addEventListener('click', () => openHelp(btn.dataset.help));
  });
  document.getElementById('btn-help-close').addEventListener('click', () => helpModal.classList.add('hidden'));
  document.getElementById('btn-help-ok').addEventListener('click', () => helpModal.classList.add('hidden'));
  helpModal.addEventListener('click', (e) => {
    if (e.target === helpModal) helpModal.classList.add('hidden');
  });

  // hash 变化 → 按路由切换（主页 / 详情）
  window.addEventListener('hashchange', () => {
    const r = parseHash();
    if (r.view === 'home') {
      showHome();
    } else if (r.view === 'stock' && r.code !== AppState.currentCode) {
      openStock(r.code);
    }
  });

  // "← 主页" 返回按钮
  document.getElementById('btn-back-home').addEventListener('click', () => {
    navigate(null);
  });


  // ═══ 初始化 ═══
  (async function init() {
    Kline.init();
    loadChips();

    // 恢复页面刷新时的训练状态
    try {
      const { jobs } = await API.activeJobs();
      (jobs || []).forEach(j => {
        AppState.activeJobs[j.id] = { type: j.type, code: j.code };
      });
    } catch (e) { /* ignore */ }

    switchList(0);  // 默认历史 tab

    // 深链路由：hash 指定股票则进详情，否则进主页
    const r = parseHash();
    if (r.view === 'stock') {
      openStock(r.code);
    } else {
      showHome();
    }
  })();
})();
