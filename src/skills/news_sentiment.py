"""skills/news_sentiment.py — 新闻舆情分析技能

从东方财富获取个股近期新闻，通过关键词匹配或 DeepSeek 分析市场情绪，
判断新闻面对股价的短期影响。

这是 9 大技能中唯一直接接入实时外部信息的技能。

优化记录：
  - _fetch_news 加入时间过滤（默认30天）和数量上限（默认30条）
  - 模块级内存缓存（5分钟TTL），同一股票短时间内不重复请求
  - 关键词匹配改为每篇文章每个关键词只计一次，避免 title+content 重复命中
  - _llm_analyze 加入 JSON 截断修复逻辑
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


# ============================================================================
# 模块级缓存（避免短时间内重复请求同一股票的新闻）
# ============================================================================
_CACHE_TTL = 300  # 缓存有效期：5 分钟
_news_cache: dict = {}  # {code: (timestamp, items)}


def _cached_fetch_news(code: str, max_items: int = 30, max_age_days: int = 30) -> list:
    """带缓存的新闻获取

    Args:
        code: 股票代码
        max_items: 最多返回条数
        max_age_days: 只保留最近 N 天的新闻
    """
    now = time.time()
    if code in _news_cache:
        ts, items = _news_cache[code]
        if now - ts < _CACHE_TTL:
            return items

    items = _fetch_news_raw(code, max_items, max_age_days)
    _news_cache[code] = (now, items)
    return items


def _fetch_news_raw(code: str, max_items: int = 30, max_age_days: int = 30) -> list:
    """从东方财富获取个股新闻（原始请求）

    Args:
        code: 股票代码（6位数字）
        max_items: 最多返回条数
        max_age_days: 只保留最近 N 天的新闻
    """
    try:
        import akshare as ak
        news_df = ak.stock_news_em(symbol=code)
        if news_df.empty:
            return []

        cutoff = datetime.now() - timedelta(days=max_age_days)

        items = []
        for _, row in news_df.iterrows():
            # 时间过滤
            time_str = str(row.get("发布时间", ""))
            try:
                news_time = pd.to_datetime(time_str)
                if news_time < cutoff:
                    continue
            except Exception:
                pass  # 时间解析失败时保留

            items.append({
                "title": str(row.get("新闻标题", "")),
                "content": str(row.get("新闻内容", ""))[:300],
                "time": time_str,
                "source": str(row.get("文章来源", "")),
            })

            if len(items) >= max_items:
                break

        return items
    except ImportError:
        return []
    except Exception:
        return []


def _repair_news_json(content: str) -> dict:
    """修复新闻 LLM 分析中被截断的 JSON"""
    open_braces = content.count("{") - content.count("}")
    open_brackets = content.count("[") - content.count("]")

    # 检查是否在字符串中间截断
    in_string = False
    for i, ch in enumerate(content):
        if ch == '"' and (i == 0 or content[i - 1] != '\\'):
            in_string = not in_string

    if in_string:
        last_comma = content.rfind(',"')
        if last_comma > 0:
            content = content[:last_comma]
        else:
            content = content.rstrip() + '"'

    content = content.rstrip(",\n\r ")
    content += "]" * open_brackets
    content += "}" * open_braces

    return json.loads(content)


class NewsSentiment(BaseSkill):
    """新闻舆情分析技能

    - 获取个股近期新闻（东方财富来源，免费）
    - 有 DeepSeek 时：LLM 分析新闻情绪和潜在影响
    - 无 DeepSeek 时：关键词匹配（利好/利空词汇表）
    - 新闻条数不足时自动降级为 hold
    """

    def __init__(self):
        super().__init__(
            name="news_sentiment",
            description="获取近期股票相关新闻，分析市场舆情和消息面影响",
            category="舆情分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        """获取新闻并分析舆情

        Args:
            df: 历史数据（本技能不使用df，仅需code）
            code: 股票代码（6位数字）
        """
        if not code or len(code) < 6:
            return SkillResult(
                skill_name=self.name, signal=SIGNAL_HOLD,
                confidence=0.0, explanation="缺少股票代码，无法查询新闻",
            )

        # 1. 获取新闻（带缓存 + 30天时间过滤）
        try:
            news_items = _cached_fetch_news(code, max_items=30, max_age_days=30)
        except Exception as e:
            return SkillResult(
                skill_name=self.name, signal=SIGNAL_HOLD,
                confidence=0.0,
                explanation=f"新闻获取失败: {e}",
            )

        if not news_items:
            return SkillResult(
                skill_name=self.name, signal=SIGNAL_HOLD,
                confidence=0.3, explanation="近期无相关新闻",
                patterns_detected=["无近期相关新闻"],
            )

        # 2. 分析新闻情绪（全局 LLM 开关关闭时绝不调用 LLM，走关键词分析）
        from src.llm_config import llm_enabled, llm_api_key, llm_api_base, llm_model
        if llm_enabled() and llm_api_key():
            return self._llm_analyze(
                news_items, code, llm_api_key(),
                api_base=llm_api_base(), model=llm_model(),
            )
        else:
            return self._keyword_analyze(news_items, code)

    # ========================================================================
    # LLM 分析（有 DeepSeek 时）
    # ========================================================================

    def _llm_analyze(self, news_items: list, code: str, api_key: str,
                     api_base: str = "https://api.deepseek.com",
                     model: str = "deepseek-v4-flash") -> SkillResult:
        """调用 DeepSeek 分析新闻情绪（api_base/model 取自全局配置）"""
        # 拼新闻摘要（最多 8 条）
        news_text_parts = []
        for i, item in enumerate(news_items[:8], 1):
            news_text_parts.append(
                f"{i}. [{item['time'][:10]}] {item['title']}\n"
                f"   {item['content'][:200]}"
            )
        news_text = "\n\n".join(news_text_parts)

        prompt = f"""你是A股舆情分析师。分析以下关于股票{code}的近期新闻，判断整体情绪对股价的影响。

新闻列表：
{news_text}

请分析：
1. 这些新闻整体是利好、利空还是中性？
2. 最重要的1-2条新闻是什么？会对股价产生多大影响？
3. 综合判断：买入/卖出/持有？把握度多少？

返回严格JSON（不要markdown代码块）：
{{"sentiment": "positive|negative|neutral", "signal": "buy|sell|hold", "confidence": 0.0-1.0, "summary": "50字以内的舆情总结", "key_news": ["最重要的利好或利空新闻标题"]}}"""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是专业A股舆情分析师，只返回JSON格式。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # JSON 截断修复
                result = _repair_news_json(content)

        except Exception:
            # LLM 失败，回退到关键词
            return self._keyword_analyze(news_items, code)

        sentiment = result.get("sentiment", "neutral")
        signal = result.get("signal", "hold")
        confidence = float(result.get("confidence", 0.5))
        summary = result.get("summary", "")
        key_news = result.get("key_news", [])

        # 构建解释
        sentiment_cn = {"positive": "偏正面", "negative": "偏负面", "neutral": "中性"}
        explanation = f"舆情{sentiment_cn.get(sentiment, '中性')}：{summary}"

        patterns = [f"新闻舆情: {sentiment_cn.get(sentiment, '')}"] + key_news[:3]

        confidence = max(0.35, min(0.85, confidence))

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=len(key_news) / 5.0,
            explanation=explanation,
            patterns_detected=patterns,
            metadata={
                "news_count": len(news_items),
                "sentiment": sentiment,
                "key_news": key_news,
                "news_items": news_items[:5],  # 保留原始新闻供后续使用
            },
        )

    # ========================================================================
    # 关键词分析（无 DeepSeek 时的回退）
    # ========================================================================

    # 利好关键词 + 权重
    _POSITIVE_WORDS = {
        "业绩大增": 3, "预增": 3, "超预期": 3, "涨停": 3,
        "中标": 2.5, "签约": 2.5, "订单": 2.5,
        "回购": 2.5, "增持": 2.5, "分红": 2,
        "利好": 2, "突破": 2, "创新高": 2.5,
        "机构买入": 2.5, "评级上调": 2.5, "推荐": 2,
        "扭亏": 2.5, "盈利": 2, "增长": 1.5,
        "合作": 1.5, "战略": 1.5, "布局": 1,
        "政策利好": 2.5, "补贴": 2, "扶持": 2,
        "产能释放": 1.5, "新品发布": 1.5, "临床试验": 2,
    }

    # 利空关键词 + 权重（负数）
    _NEGATIVE_WORDS = {
        "亏损": -3, "预亏": -3, "下降": -2, "下滑": -2.5,
        "减持": -2.5, "套现": -3, "跌停": -3,
        "退市": -4, "ST": -3, "停牌": -2.5,
        "处罚": -3, "罚款": -3, "调查": -3, "诉讼": -2.5,
        "评级下调": -2.5, "卖出评级": -2.5,
        "业绩不及": -3, "低于预期": -3, "暴雷": -3.5,
        "资金困难": -3, "债务": -2.5, "违约": -3.5,
        "商誉减值": -3, "计提": -2, "坏账": -2.5,
        "大股东": -1.5, "质押": -2, "冻结": -2.5,
        "监管": -2, "问询": -2, "警示": -2.5,
    }

    def _keyword_analyze(self, news_items: list, code: str) -> SkillResult:
        """基于关键词匹配的新闻情绪分析

        每篇文章每个关键词只计一次分（用 set 去重），避免同篇文章内
        title 和 content 重复命中同一关键词导致分数膨胀。
        """
        score = 0.0
        pos_hits = []
        neg_hits = []

        for item in news_items:
            title = item.get("title", "")
            content = item.get("content", "")
            text = title + content

            # 用 set 记录本条新闻已经命中的关键词，每词只计一次
            seen_pos = set()
            seen_neg = set()

            for word, weight in self._POSITIVE_WORDS.items():
                if word in text and word not in seen_pos:
                    score += weight
                    pos_hits.append(word)
                    seen_pos.add(word)

            for word, weight in self._NEGATIVE_WORDS.items():
                if word in text and word not in seen_neg:
                    score += weight  # weight is negative
                    neg_hits.append(word)
                    seen_neg.add(word)

        # 归一化
        n = len(news_items)
        if n > 0:
            score = score / n

        # 判定
        if score >= 1.0:
            signal = SIGNAL_BUY
            confidence = min(0.40 + score * 0.12, 0.85)
        elif score <= -1.0:
            signal = SIGNAL_SELL
            confidence = min(0.40 + abs(score) * 0.12, 0.85)
        else:
            signal = SIGNAL_HOLD
            confidence = 0.40

        patterns = []
        if pos_hits:
            patterns.append(f"利好信号: {', '.join(list(set(pos_hits))[:5])}")
        if neg_hits:
            patterns.append(f"利空信号: {', '.join(list(set(neg_hits))[:5])}")
        if not patterns:
            patterns.append("新闻面偏中性，无明显利多利空")

        sentiment_score = f"+{score:.1f}" if score > 0 else f"{score:.1f}"
        explanation = (
            f"舆情关键词分析（{len(news_items)}条新闻，得分{sentiment_score}）："
            + ("偏正面" if score > 0.5 else ("偏负面" if score < -0.5 else "中性"))
        )

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=min(abs(score) / 5.0, 1.0),
            explanation=explanation,
            patterns_detected=patterns,
            metadata={"news_count": len(news_items), "score": score, "news_items": news_items[:5]},
        )
