/* kline.js — ECharts K 线图：candlestick + MA5/20/60 + 成交量副图 */
(function (window) {
  'use strict';

  let chart = null;

  function init() {
    const el = document.getElementById('kline');
    if (!el) return null;
    // 首次 init 前容器可能处于 hidden 状态（detail-view display:none），
    // ECharts 会拿到 0 宽度。用 ResizeObserver 监听容器尺寸变化自动 resize，
    // 解决图表"缩在最左边很窄区域"、需手动调窗口才恢复的 bug。
    if (!chart) {
      chart = echarts.init(el);
      if (typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => {
          if (chart) chart.resize();
        });
        ro.observe(el);
      }
    }
    window.addEventListener('resize', () => chart && chart.resize());
    return chart;
  }

  function resize() {
    if (chart) chart.resize();
  }

  function render(data) {
    if (!chart) chart = init();
    if (!chart) return;

    const dates = data.dates || [];
    const candles = data.candles || [];
    const volumes = data.volumes || [];
    const volumeDirs = data.volume_dirs || [];
    const turnover = data.turnover_rates || [];

    // 成交量柱颜色（A股惯例：红涨绿跌）
    const volColors = volumes.map((_, i) =>
      (volumeDirs[i] || 1) === 1 ? 'rgba(248,81,73,0.5)' : 'rgba(63,185,80,0.5)'
    );
    // 换手率线
    const turnoverLine = turnover.map((v, i) =>
      (v == null ? null : +(v * 100).toFixed(2))
    );

    // 默认展示最近 200 天，滑杆 / 滚轮可回看更早（后端返回约 500 天两年数据）
    const DEFAULT_VIEW_DAYS = 200;
    const viewStart = Math.max(0, 100 - (DEFAULT_VIEW_DAYS / Math.max(dates.length, 1)) * 100);

    const option = {
      backgroundColor: 'transparent',
      animation: false,
      legend: {
        data: ['K线', 'MA5', 'MA20', 'MA60', '成交量', '换手率%'],
        textStyle: { color: '#8b949e', fontSize: 11 },
        top: 4,
        left: 8,
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        textStyle: { color: '#e6edf3', fontSize: 12 },
        formatter: function (params) {
          const i = params[0].dataIndex;
          const c = candles[i];
          const d = dates[i];
          let s = `<div style="font-weight:700;margin-bottom:4px">${d}</div>`;
          if (c) {
            const open = c[0], close = c[1], low = c[2], high = c[3];
            const color = close >= open ? '#f85149' : '#3fb950';  // A股惯例：红涨绿跌
            s += `<div>开盘 <span style="color:${color}">${open}</span></div>`;
            s += `<div>收盘 <span style="color:${color}">${close}</span></div>`;
            s += `<div>最低 <span style="color:${color}">${low}</span></div>`;
            s += `<div>最高 <span style="color:${color}">${high}</span></div>`;
          }
          if (volumes[i] != null) s += `<div>成交量 ${(+volumes[i]).toLocaleString()}</div>`;
          if (turnover[i] != null) s += `<div>换手率 ${(turnover[i] * 100).toFixed(2)}%</div>`;
          return s;
        },
      },
      axisPointer: { link: [{ xAxisIndex: 'all' }] },
      grid: [
        { left: 56, right: 56, top: 34, height: '55%' },
        { left: 56, right: 56, top: '72%', height: '18%' },
      ],
      xAxis: [
        { type: 'category', data: dates, boundaryGap: true, axisLine: { lineStyle: { color: '#30363d' } }, axisLabel: { color: '#8b949e', fontSize: 10 }, axisPointer: { label: { backgroundColor: '#30363d' } } },
        { type: 'category', gridIndex: 1, data: dates, axisLine: { lineStyle: { color: '#30363d' } }, axisLabel: { show: false }, splitLine: { show: false } },
      ],
      yAxis: [
        { scale: true, splitLine: { lineStyle: { color: '#21262d' } }, axisLabel: { color: '#8b949e', fontSize: 10 } },
        { gridIndex: 1, splitLine: { show: false }, axisLabel: { color: '#8b949e', fontSize: 10 } },
      ],
      dataZoom: [
        { type: 'inside', xAxisIndex: [0, 1], start: viewStart, end: 100 },
        { type: 'slider', xAxisIndex: [0, 1], start: viewStart, end: 100, bottom: 4, height: 16, borderColor: '#30363d', backgroundColor: '#161b22', fillerColor: 'rgba(88,166,255,0.15)', textStyle: { color: '#8b949e', fontSize: 10 } },
      ],
      series: [
        {
          name: 'K线', type: 'candlestick', data: candles,
          itemStyle: {
            color: '#f85149', color0: '#3fb950',   // 阳线=红 阴线=绿（A股惯例）
            borderColor: '#f85149', borderColor0: '#3fb950',
          },
        },
        { name: 'MA5', type: 'line', data: data.ma5 || [], smooth: true, showSymbol: false, connectNulls: true, lineStyle: { width: 1, color: '#d29922' } },
        { name: 'MA20', type: 'line', data: data.ma20 || [], smooth: true, showSymbol: false, connectNulls: true, lineStyle: { width: 1, color: '#58a6ff' } },
        { name: 'MA60', type: 'line', data: data.ma60 || [], smooth: true, showSymbol: false, connectNulls: true, lineStyle: { width: 1, color: '#bc8cff' } },
        {
          name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: volumes,
          itemStyle: { color: (p) => volColors[p.dataIndex] },
        },
        {
          name: '换手率%', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: turnoverLine,
          showSymbol: false, connectNulls: true, lineStyle: { width: 1, color: '#3fb950' },
        },
      ],
    };

    chart.setOption(option, true);
    chart.hideLoading();
  }

  function showLoading() {
    if (!chart) chart = init();
    if (chart) chart.showLoading({ text: '加载中…', color: '#58a6ff', maskColor: 'rgba(13,17,23,0.6)' });
  }

  window.Kline = { init, resize, render, showLoading };
})(window);
