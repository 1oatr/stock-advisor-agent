/* cards.js — 实时指标卡 + 5 张分析卡片渲染 */
(function (window) {
  'use strict';

  // ── 信号样式 ──
  function sigClass(signal) {
    return signal === 'buy' ? 'signal-buy'
      : signal === 'sell' ? 'signal-sell'
      : 'signal-hold';
  }
  function sigText(signal) {
    return signal === 'buy' ? '增持' : signal === 'sell' ? '减持' : '持有';
  }
  function pctClass(v) { return v >= 0 ? 'up' : 'down'; }
  function fmtNum(v, digits = 2) {
    if (v == null || isNaN(v)) return '--';
    return Number(v).toFixed(digits);
  }
  function fmtBig(v) {
    if (v == null || isNaN(v)) return '--';
    const n = Number(v);
    if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(2) + '万亿';
    if (Math.abs(n) >= 1e8) return (n / 1e8).toFixed(2) + '亿';
    if (Math.abs(n) >= 1e4) return (n / 1e4).toFixed(2) + '万';
    return n.toFixed(0);
  }

  // ── 实时指标卡 ──
  function renderQuote(data) {
    const box = document.getElementById('quote-cards');
    const rt = (data && data.realtime) || {};
    const ff = (data && data.fund_flow) || {};
    const info = (data && data.info) || {};
    const isRealtime = data && data.time !== '日线近似';
    const market = data && data.market || 'a';
    const isA = market === 'a';

    const cards = [
      { label: '现价', value: fmtNum(rt.price), cls: pctClass(rt.pct), sub: `${rt.pct >= 0 ? '+' : ''}${fmtNum(rt.pct)}%` },
      { label: '今开', value: fmtNum(rt.open) },
      { label: '最高', value: fmtNum(rt.high) },
      { label: '最低', value: fmtNum(rt.low) },
      { label: '昨收', value: fmtNum(rt.prev_close) },
      { label: '成交额', value: fmtBig(rt.amount) },
    ];
    // A股补充完整指标；港股/美股仅显示基础字段
    if (isA) {
      cards.push(
        { label: '换手率', value: rt.turnover_rate != null ? fmtNum(rt.turnover_rate) : '--', sub: '%' },
        { label: '量比', value: fmtNum(rt.volume_ratio) },
        { label: '市盈率(动)', value: fmtNum(rt.pe_dynamic) },
        { label: '市净率', value: fmtNum(rt.pb) },
        { label: '总市值', value: fmtBig(rt.total_mv) },
        { label: '流通市值', value: fmtBig(rt.circ_mv) },
        { label: '外盘(买)', value: fmtBig(rt.outer_vol) },
        { label: '内盘(卖)', value: fmtBig(rt.inner_vol) },
        { label: '主力净流入', value: fmtBig(ff.main_net_inflow), cls: ff.main_net_inflow > 0 ? 'up' : ff.main_net_inflow < 0 ? 'down' : '' },
        { label: '行业', value: info.industry || '--' },
      );
    }

    box.innerHTML = cards.map(c => `
      <div class="q-card">
        <div class="q-label">${c.label}</div>
        <div class="q-value ${c.cls || ''}">${c.value}${c.sub ? `<span class="q-sub">${c.sub}</span>` : ''}</div>
      </div>
    `).join('');

    if (!isRealtime) {
      const note = document.createElement('div');
      note.className = 'empty-hint';
      note.style.padding = '4px';
      note.textContent = isA
        ? '⚠️ 实时数据暂不可用（东财接口受限），已显示日线近似'
        : '⚠️ 该市场实时行情/财务指标数据源受限，已显示日线近似';
      box.appendChild(note);
    }
  }

  // ── 规则引擎卡片 ──
  function renderRulesCard(result) {
    const body = document.querySelector('[data-card="rules"] .card-body');
    if (!result || result.error) {
      body.className = 'card-body';
      body.textContent = result && result.error ? `❌ ${result.error}` : '无数据';
      return;
    }
    const rule = result.rule_engine || {};
    const sig = rule.composite_signal || 'hold';
    const strength = rule.composite_strength || 0;

    let html = `<div class="sig-line">综合: <span class="${sigClass(sig)}">${sigText(sig)}</span> (强度 ${fmtNum(strength * 100, 1)}%)</div>`;
    html += `<div class="conf-bar"><div class="conf-fill" style="width:${Math.round(strength * 100)}%;background:${sig === 'buy' ? '#f85149' : sig === 'sell' ? '#3fb950' : '#d29922'}"></div></div>`;

    const topRules = rule.top_rules || [];
    if (topRules.length) {
      html += topRules.slice(0, 4).map(r => {
        const rc = r.signal === 'buy' ? 'signal-buy' : r.signal === 'sell' ? 'signal-sell' : 'signal-hold';
        return `<div class="sig-line"><span class="sig-name">${r.name}:</span> <span class="${rc}">${sigText(r.signal)}</span> <span style="color:#8b949e">${r.explanation || ''}</span></div>`;
      }).join('');
    }
    body.className = 'card-body';
    body.innerHTML = html;
  }

  // ── 技能分析卡片 ──
  function renderSkillsCard(result) {
    const body = document.querySelector('[data-card="skills"] .card-body');
    if (!result || result.error) {
      body.className = 'card-body';
      body.textContent = result && result.error ? `❌ ${result.error}` : '无数据';
      return;
    }
    const action = result.action || 'hold';
    const conf = result.confidence || 0;
    let html = `<div class="sig-line">信号: <span class="${sigClass(action)}">${sigText(action)}</span> (置信度 ${fmtNum(conf * 100, 1)}%)</div>`;
    html += `<div class="conf-bar"><div class="conf-fill" style="width:${Math.round(conf * 100)}%;background:${action === 'buy' ? '#f85149' : action === 'sell' ? '#3fb950' : '#d29922'}"></div></div>`;
    if (result.analysis_text) html += `<div>${result.analysis_text.slice(0, 180)}</div>`;
    const signals = result.key_signals || [];
    if (signals.length) html += `<div style="color:#8b949e;margin-top:4px">关键: ${signals.slice(0, 3).join('、')}</div>`;
    body.className = 'card-body';
    body.innerHTML = html;
  }

  // ── LLM 分析卡片 ──
  function renderLlmCard(result) {
    const body = document.querySelector('[data-card="llm"] .card-body');
    if (!result) {
      body.className = 'card-body';
      body.textContent = '点击上方"LLM 分析"按钮触发';
      return;
    }
    if (result.error) {
      body.className = 'card-body';
      body.textContent = `❌ ${result.error}`;
      return;
    }
    if (result.disabled) {
      body.className = 'card-body';
      body.textContent = '🔒 LLM 已关闭（未设置 API Key 或手动关闭）';
      return;
    }
    const action = result.action || 'hold';
    const conf = result.confidence || 0;
    let html = `<div class="sig-line">信号: <span class="${sigClass(action)}">${sigText(action)}</span> (置信度 ${fmtNum(conf * 100, 1)}%)</div>`;
    html += `<div class="conf-bar"><div class="conf-fill" style="width:${Math.round(conf * 100)}%;background:${action === 'buy' ? '#f85149' : action === 'sell' ? '#3fb950' : '#d29922'}"></div></div>`;
    if (result.analysis_text) html += `<div>${result.analysis_text.slice(0, 180)}</div>`;
    if (result.risk_note) html += `<div style="color:#d29922;margin-top:4px">⚠️ ${result.risk_note.slice(0, 120)}</div>`;
    body.className = 'card-body';
    body.innerHTML = html;
  }

  // ── RL 分析卡片 ──
  function renderRlCard(result) {
    const body = document.querySelector('[data-card="rl"] .card-body');
    if (!result || result.error) {
      body.className = 'card-body';
      body.textContent = result && result.error ? `❌ ${result.error}` : '无数据';
      return;
    }
    if (result.untrained) {
      body.className = 'card-body';
      const reason = result.reason || '';
      body.textContent = reason.includes('港股') || reason.includes('美股')
        ? '🚫 RL 暂不支持港股/美股'
        : '⏳ RL 模型未训练。点击"RL 训练"按钮训练后获取预测。';
      return;
    }
    const action = result.action || 'hold';
    const conf = result.confidence || 0;
    const fresh = result.model_fresh;
    const d = result.details || {};
    let html = `<div class="sig-line">信号: <span class="${sigClass(action)}">${sigText(action)}</span> (置信度 ${fmtNum(conf * 100, 1)}%)</div>`;
    html += `<div class="conf-bar"><div class="conf-fill" style="width:${Math.round(conf * 100)}%;background:${action === 'buy' ? '#f85149' : action === 'sell' ? '#3fb950' : '#d29922'}"></div></div>`;
    html += `<div style="color:#8b949e">新鲜度: ${fresh ? '✅' : '⚠️'} ${fresh ? '已训练' : '模型较旧'}</div>`;
    if (d.total_steps) {
      html += `<div style="color:#8b949e">买${(d.buy_ratio * 100).toFixed(0)}% 卖${(d.sell_ratio * 100).toFixed(0)}% 持${(d.hold_ratio * 100).toFixed(0)}%</div>`;
    }
    body.className = 'card-body';
    body.innerHTML = html;
  }

  // ── 三路融合综合建议卡片 ──
  function renderFusedCard(result) {
    const body = document.querySelector('[data-card="fused"] .card-body');
    if (!result) {
      body.className = 'card-body';
      body.textContent = '点击"三路预测"按钮触发';
      return;
    }
    if (result.error) {
      body.className = 'card-body';
      body.textContent = `❌ ${result.error}`;
      return;
    }
    const recs = result.recommendations || [];
    const market = result.market_state || 'unknown';
    const marketIcon = { bull: '🚀', bear: '📉', range: '📊' };
    let html = `<div class="sig-line">大盘: ${marketIcon[market] || '❓'} ${market} | 建议仓位 ${result.suggested_position_pct ?? '--'}%</div><hr style="border-color:#30363d;margin:6px 0">`;

    recs.forEach(rec => {
      if (rec.error) { html += `<div class="sig-line">❌ ${rec.code}: ${rec.error}</div>`; return; }
      const fused = rec.fused || {};
      const f = fused.action || 'hold';
      const pos = fused.position || 0;
      const consensus = fused.consensus || 'low';
      const consText = { high: '三路一致 ✅', medium: '两路一致 ⚡', low: '三路分歧 ⚠️' }[consensus] || consensus;
      html += `<div style="font-weight:700;margin-top:6px">${rec.code}</div>`;
      html += `<div class="sig-line">综合: <span class="${sigClass(f)}">${sigText(f)}</span> (${fmtNum(fused.confidence * 100, 1)}%)</div>`;
      html += `<div class="conf-bar"><div class="conf-fill" style="width:${Math.round((fused.confidence || 0) * 100)}%;background:${f === 'buy' ? '#f85149' : f === 'sell' ? '#3fb950' : '#d29922'}"></div></div>`;
      html += `<div style="color:#8b949e">仓位 ${(pos * 100).toFixed(1)}% | ${consText}</div>`;
      const llm = rec.llm_skills || rec.llm_analysis;
      const rules = rec.rules || {};
      const rl = rec.rl || {};
      html += `<div class="sig-line" style="font-size:11px;color:#8b949e">`;
      html += `LLM:${sigText((llm || {}).action || 'hold')} 规则:${sigText(rules.signal || 'hold')} RL:${rl.untrained ? '未训练' : sigText(rl.action || 'hold')}`;
      html += `</div>`;
    });

    const ranking = result.ranking || [];
    if (ranking.length) {
      html += `<hr style="border-color:#30363d;margin:6px 0">`;
      html += ranking.map(r => `<div style="font-size:11px;color:#8b949e">${r.rank}. ${r.code} → ${sigText(r.action)} (${fmtNum(r.confidence * 100, 1)}%)</div>`).join('');
    }
    body.className = 'card-body';
    body.innerHTML = html;
  }

  // ── 分析卡片统一入口 ──
  function renderCard(name, result) {
    if (name === 'rules') renderRulesCard(result);
    else if (name === 'skills') renderSkillsCard(result);
    else if (name === 'llm') renderLlmCard(result);
    else if (name === 'rl') renderRlCard(result);
    else if (name === 'fused') renderFusedCard(result);
  }

  function setCardLoading(name, text) {
    const body = document.querySelector(`[data-card="${name}"] .card-body`);
    if (body) { body.className = 'card-body loading'; body.textContent = text || '加载中…'; }
  }

  window.Cards = {
    renderQuote, renderCard, setCardLoading,
    sigClass, sigText, fmtNum, fmtBig, pctClass,
  };
})(window);
