/* api.js — fetch 封装：统一错误处理 / 超时 / JSON 解析 + localStorage TTL 缓存 */
(function (window) {
  'use strict';

  // ── localStorage TTL 缓存（避免每次点击股票重复请求后端） ──
  const CACHE_PREFIX = 'sa_cache_v1:';
  function cacheGet(key) {
    try {
      const raw = localStorage.getItem(CACHE_PREFIX + key);
      if (!raw) return null;
      const { t, ttl, data } = JSON.parse(raw);
      if (Date.now() - t > ttl) {
        localStorage.removeItem(CACHE_PREFIX + key);
        return null;
      }
      return data;
    } catch (e) { return null; }
  }
  function cacheSet(key, data, ttl) {
    try {
      localStorage.setItem(CACHE_PREFIX + key,
        JSON.stringify({ t: Date.now(), ttl, data }));
    } catch (e) { /* localStorage 超限等，忽略 */ }
  }
  function clearCache() {
    try {
      const keys = [];
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith(CACHE_PREFIX)) keys.push(k);
      }
      keys.forEach(k => localStorage.removeItem(k));
    } catch (e) { /* ignore */ }
  }

  async function _fetch(path, options = {}) {
    const { timeout = 30000, cacheTtl = 0, ...rest } = options;
    const isGet = !rest.method || rest.method === 'GET';
    if (isGet && cacheTtl > 0) {
      const hit = cacheGet(path);
      if (hit != null) return hit;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(path, { ...rest, signal: controller.signal });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const err = new Error(data.error || `HTTP ${res.status}`);
        err.code = data.code || res.status;
        throw err;
      }
      if (isGet && cacheTtl > 0) cacheSet(path, data, cacheTtl);
      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  const API = {
    get: (path, params = {}, cacheTtl = 0, timeout = 30000) => {
      const qs = new URLSearchParams(params).toString();
      return _fetch('/api' + path + (qs ? '?' + qs : ''), { cacheTtl, timeout });
    },
    post: (path, body) =>
      _fetch('/api' + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
    put: (path, body) =>
      _fetch('/api' + path, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
    del: (path) => _fetch('/api' + path, { method: 'DELETE' }),

    // ── 具体接口（cacheTtl 单位: 秒；0 = 不缓存） ──
    health: () => API.get('/health'),
    marketState: () => API.get('/market/state', {}, 60),
    llmStatus: () => API.get('/llm/status', {}, 15),
    search: (q, limit = 10) => API.get('/stocks/search', { q, limit }, 30),
    kline: (code, days = 500) => API.get(`/stocks/${code}/kline`, { days }, 300),
    quote: (code) => API.get(`/stocks/${code}/quote`, {}, 30),
    analyze: (code) => API.get(`/stocks/${code}/analyze`, {}, 600),
    skills: (code) => API.get(`/stocks/${code}/skills`, {}, 600),
    llm: (code) => API.get(`/stocks/${code}/llm`, {}, 0, 90000),
    rl: (code, refresh = false) => API.get(
      `/stocks/${code}/rl`, refresh ? { refresh: 1 } : {}, refresh ? 0 : 600),

    marketList: (market, q = '', limit = 20) => API.get(`/market/${market}/stocks`, { q, limit }, 60),
    history: (limit = 50) => API.get('/history', { limit }, 5),
    watchlist: () => API.get('/watchlist', {}, 5),
    addWatch: (code) => API.post('/watchlist', { code }),
    removeWatch: (code) => API.del(`/watchlist/${code}`),

    submitJob: (type, params) => API.post('/jobs', { type, ...params }),
    jobStatus: (id) => API.get(`/jobs/${id}`),
    cancelJob: (id) => API.post(`/jobs/${id}/cancel`),
    activeJobs: () => API.get('/jobs/active'),

    // 清空全部前端缓存（规则/技能参数保存后调用，强制重新拉取）
    clearCache,
  };

  window.API = API;
})(window);
