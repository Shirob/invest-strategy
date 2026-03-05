"""A-share investment ETL and prompt generation scaffold.

This script fetches market data via akshare, aggregates macro/industry news,
collects user position/emotion status from CLI, and renders a standardized
Markdown report for LLM-based discipline checks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

import akshare as ak
import pandas as pd
import requests


TICKER_POOL: dict[str, str] = {
    "中际旭创": "300308",
    "紫金矿业": "601899",
    "中国海油": "600938",
}

KEYWORDS: tuple[str, ...] = (
    "光模块",
    "算力",
    "英伟达",
    "铜",
    "铝",
    "原油",
    "美联储",
    "非农",
)

RSS_SOURCES: tuple[str, ...] = (
    "https://www.jin10.com/rss",
    "https://rsshub.app/cls/telegraph",
)


@dataclass
class MarketSnapshot:
    """Normalized market snapshot for one ticker."""

    name: str
    symbol: str
    date: dt.date
    close: float
    pct_change: float
    volume: float
    ma20: float
    ma20_deviation_pct: float


@dataclass
class NewsItem:
    """News entry from RSS/crawler sources."""

    title: str
    summary: str
    published_at: Optional[dt.datetime]
    source: str


@dataclass
class UserStatus:
    """User portfolio status + emotion input."""

    ai: float
    metals: float
    oil: float
    cash: float
    emotion: str


class MarketDataFetcher:
    """Fetch A-share market data and calculate MA20 deviation."""

    def fetch_snapshot(self, name: str, symbol: str) -> MarketSnapshot:
        """Fetch latest OHLCV-based data for a ticker from akshare.

        Args:
            name: Display name in Chinese.
            symbol: Numeric A-share code, e.g. "300308".

        Returns:
            MarketSnapshot: standardized output with MA20 and deviation.

        Raises:
            RuntimeError: If data is unavailable or malformed.
        """
        try:
            hist_df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=(dt.date.today() - dt.timedelta(days=120)).strftime("%Y%m%d"),
                end_date=dt.date.today().strftime("%Y%m%d"),
                adjust="qfq",
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"获取 {name}({symbol}) 行情失败: {exc}") from exc

        if hist_df.empty or "收盘" not in hist_df.columns:
            raise RuntimeError(f"{name}({symbol}) 行情为空或字段缺失")

        df = hist_df.copy()
        df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
        df["涨跌幅"] = pd.to_numeric(df.get("涨跌幅"), errors="coerce")
        df["成交量"] = pd.to_numeric(df.get("成交量"), errors="coerce")
        df["MA20"] = df["收盘"].rolling(window=20).mean()
        latest = df.dropna(subset=["收盘", "MA20"]).iloc[-1]

        close = float(latest["收盘"])
        ma20 = float(latest["MA20"])
        pct_change = float(latest["涨跌幅"]) if pd.notna(latest["涨跌幅"]) else 0.0
        volume = float(latest["成交量"]) if pd.notna(latest["成交量"]) else 0.0
        deviation = ((close - ma20) / ma20) * 100 if ma20 else 0.0

        row_date = pd.to_datetime(latest["日期"]).date() if "日期" in latest else dt.date.today()

        return MarketSnapshot(
            name=name,
            symbol=symbol,
            date=row_date,
            close=close,
            pct_change=pct_change,
            volume=volume,
            ma20=ma20,
            ma20_deviation_pct=deviation,
        )

    def fetch_all(self, ticker_pool: dict[str, str]) -> list[MarketSnapshot]:
        """Fetch all target tickers, skip failed ones but keep process alive."""
        snapshots: list[MarketSnapshot] = []
        for name, symbol in ticker_pool.items():
            try:
                snapshots.append(self.fetch_snapshot(name=name, symbol=symbol))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {exc}", file=sys.stderr)
        return snapshots


class NewsMacroAggregator:
    """Aggregate filtered macro/industry news from RSS sources."""

    def __init__(self, sources: Iterable[str], keywords: Iterable[str]) -> None:
        self.sources = tuple(sources)
        self.keywords = tuple(keywords)

    def _parse_datetime(self, raw: str) -> Optional[dt.datetime]:
        """Parse RFC/ISO style date text from RSS nodes."""
        raw = (raw or "").strip()
        if not raw:
            return None

        candidates = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
        ]
        for fmt in candidates:
            try:
                parsed = dt.datetime.strptime(raw, fmt)
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
                return parsed
            except ValueError:
                continue
        return None

    def _contains_keyword(self, text: str) -> bool:
        return any(k in text for k in self.keywords)

    def _fetch_source(self, url: str, timeout: int = 10) -> list[NewsItem]:
        """Fetch and parse one RSS source."""
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 资讯源请求失败 {url}: {exc}", file=sys.stderr)
            return []

        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            print(f"[WARN] RSS 解析失败 {url}: {exc}", file=sys.stderr)
            return []

        news_items: list[NewsItem] = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            summary = (item.findtext("description") or item.findtext("content:encoded") or "").strip()
            pub_text = item.findtext("pubDate") or item.findtext("updated") or ""
            pub_dt = self._parse_datetime(pub_text)

            full_text = f"{title}\n{summary}"
            if self._contains_keyword(full_text):
                news_items.append(
                    NewsItem(
                        title=title,
                        summary=summary,
                        published_at=pub_dt,
                        source=url,
                    )
                )
        return news_items

    def denoise_and_summarize(self, items: list[NewsItem]) -> list[NewsItem]:
        """Placeholder for future local-LLM based noise reduction.

        Currently deduplicates by title.
        """
        seen: set[str] = set()
        cleaned: list[NewsItem] = []
        for item in items:
            if item.title and item.title not in seen:
                seen.add(item.title)
                cleaned.append(item)
        return cleaned

    def collect_last_24h(self) -> list[NewsItem]:
        """Collect keyword-matched news from the past 24 hours."""
        all_items: list[NewsItem] = []
        for source in self.sources:
            all_items.extend(self._fetch_source(source))

        cutoff = dt.datetime.utcnow() - dt.timedelta(hours=24)
        filtered = [
            n
            for n in all_items
            if (n.published_at is None or n.published_at >= cutoff)
        ]
        filtered.sort(key=lambda x: x.published_at or dt.datetime.min, reverse=True)
        return self.denoise_and_summarize(filtered)


class UserStatusTracker:
    """Interactive CLI + state persistence via SQLite and CSV fallback."""

    def __init__(self, sqlite_path: Path = Path("user_status.db"), csv_path: Path = Path("user_status_history.csv")) -> None:
        self.sqlite_path = sqlite_path
        self.csv_path = csv_path
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_status (
                    date TEXT PRIMARY KEY,
                    ai REAL NOT NULL,
                    metals REAL NOT NULL,
                    oil REAL NOT NULL,
                    cash REAL NOT NULL,
                    emotion TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _load_latest_before_today(self) -> Optional[UserStatus]:
        today = dt.date.today().isoformat()
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute(
                """
                SELECT ai, metals, oil, cash, emotion
                FROM daily_status
                WHERE date < ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (today,),
            ).fetchone()
        if row:
            return UserStatus(*map(float, row[:4]), emotion=row[4])

        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            if not df.empty:
                last = df.iloc[-1]
                return UserStatus(
                    ai=float(last.get("ai", 0.0)),
                    metals=float(last.get("metals", 0.0)),
                    oil=float(last.get("oil", 0.0)),
                    cash=float(last.get("cash", 100.0)),
                    emotion=str(last.get("emotion", "")),
                )
        return None

    def _input_percent(self, prompt: str, default: float) -> float:
        while True:
            raw = input(f"{prompt} [{default:.2f}%]: ").strip()
            if not raw:
                return default
            try:
                val = float(raw)
                if 0 <= val <= 100:
                    return val
            except ValueError:
                pass
            print("请输入 0-100 的数字。")

    def collect_cli(self) -> UserStatus:
        """Collect user status from CLI with yesterday defaults."""
        default = self._load_latest_before_today() or UserStatus(25.0, 25.0, 25.0, 25.0, "")
        print("\n--- 请输入今日仓位与主观情绪（回车采用昨日值）---")

        ai = self._input_percent("AI 仓位", default.ai)
        metals = self._input_percent("有色仓位", default.metals)
        oil = self._input_percent("石油仓位", default.oil)
        cash = self._input_percent("现金仓位", default.cash)
        emotion = input(f"今日主观情绪/冲动 [{default.emotion or '无'}]: ").strip() or default.emotion or "无"

        status = UserStatus(ai=ai, metals=metals, oil=oil, cash=cash, emotion=emotion)
        self.save_today(status)
        return status

    def save_today(self, status: UserStatus) -> None:
        """Persist today's user status to SQLite and append CSV history."""
        today = dt.date.today().isoformat()
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT INTO daily_status(date, ai, metals, oil, cash, emotion)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date)
                DO UPDATE SET
                    ai=excluded.ai,
                    metals=excluded.metals,
                    oil=excluded.oil,
                    cash=excluded.cash,
                    emotion=excluded.emotion
                """,
                (today, status.ai, status.metals, status.oil, status.cash, status.emotion),
            )
            conn.commit()

        row = pd.DataFrame(
            [
                {
                    "date": today,
                    "ai": status.ai,
                    "metals": status.metals,
                    "oil": status.oil,
                    "cash": status.cash,
                    "emotion": status.emotion,
                }
            ]
        )
        if self.csv_path.exists():
            row.to_csv(self.csv_path, mode="a", index=False, header=False)
        else:
            row.to_csv(self.csv_path, index=False)


class TemplateGenerator:
    """Render and persist markdown prompt text."""

    def generate(
        self,
        snapshots: list[MarketSnapshot],
        news: list[NewsItem],
        status: UserStatus,
        market_mood: str,
    ) -> str:
        """Build markdown prompt with fixed structure."""
        today_zh = dt.date.today().strftime("%Y年%-m月%-d日")
        snapshot_map = {s.name: s for s in snapshots}

        def _line(name: str) -> str:
            s = snapshot_map.get(name)
            if not s:
                return f"  - {name}：数据获取失败。"
            return (
                f"  - {name}：今日涨跌幅 {s.pct_change:.2f}%，"
                f"当前偏离 20 日均线 {s.ma20_deviation_pct:.2f}%。"
            )

        news_block = "\n".join(
            [
                f"- [{(n.published_at.strftime('%m-%d %H:%M') if n.published_at else '时间未知')}] {n.title}"
                + (f"：{n.summary}" if n.summary else "")
                for n in news
            ]
        ) or "- 过去 24 小时内未检索到命中关键字的资讯。"

        return f"""**【日期】**：{today_zh}
**【盘面】**：
- 大盘情绪：{market_mood}
- 核心标的：
{_line('中际旭创')}
{_line('紫金矿业')}
{_line('中国海油')}
**【资讯】**：
{news_block}
**【我的状态】**：
- 当前仓位：AI {status.ai:.2f}%，有色 {status.metals:.2f}%，石油 {status.oil:.2f}%，现金 {status.cash:.2f}%。
- 我的主观冲动：{status.emotion}

请作为理性的投资机器，基于“拉开距离，控制仓位”的八字原则，给出今日策略分析与操作建议。
"""

    def save_report(self, content: str, output_dir: Path = Path(".")) -> Path:
        """Save report to YYYY-MM-DD_report.txt."""
        filename = f"{dt.date.today().isoformat()}_report.txt"
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="A股投资 ETL 与 Prompt 生成脚手架")
    parser.add_argument(
        "--market-mood",
        default="（待补充：可手动输入或后续接入自动摘要）",
        help="大盘情绪概述",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="非交互模式：使用昨日仓位与空情绪",
    )
    return parser.parse_args()


def main() -> int:
    """Program entrypoint."""
    args = parse_args()

    market_fetcher = MarketDataFetcher()
    news_aggregator = NewsMacroAggregator(sources=RSS_SOURCES, keywords=KEYWORDS)
    user_tracker = UserStatusTracker()
    template_gen = TemplateGenerator()

    snapshots = market_fetcher.fetch_all(TICKER_POOL)
    news = news_aggregator.collect_last_24h()

    if args.non_interactive:
        status = user_tracker._load_latest_before_today() or UserStatus(25.0, 25.0, 25.0, 25.0, "无")
        user_tracker.save_today(status)
    else:
        status = user_tracker.collect_cli()

    report = template_gen.generate(
        snapshots=snapshots,
        news=news,
        status=status,
        market_mood=args.market_mood,
    )

    print("\n===== 今日投资纪律检查 Prompt =====\n")
    print(report)

    save_path = template_gen.save_report(report)
    print(f"\n报告已保存到: {save_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
