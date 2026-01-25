"""SQLite persistence helpers for trading operations."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

from .models import Operation


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _initialize_database(self) -> None:
        with self._connect() as con:
            cur = con.cursor()
            # Original operations table
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                tipo TEXT,
                entrada REAL,
                entrada_min REAL,
                entrada_max REAL,
                stop REAL,
                alvo REAL,
                parcial_preco REAL,
                parcial_pontos REAL,
                quantidade INTEGER,
                observacoes TEXT,
                preco_atual REAL,
                pontos_alvo REAL,
                pontos_stop REAL,
                tick_size REAL,
                risco_retorno REAL,
                status TEXT,
                created_at TEXT,
                pdf_path TEXT,
                timeframe TEXT,
                indicators TEXT
            )
            """
            )
            
            # Swing Trade table
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS swing_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry REAL NOT NULL,
                entry_min REAL NOT NULL,
                entry_max REAL NOT NULL,
                target REAL NOT NULL,
                stop REAL NOT NULL,
                quantity INTEGER NOT NULL,
                trade_date TEXT NOT NULL,
                timeframe_major TEXT DEFAULT '1d',
                timeframe_minor TEXT DEFAULT '1h',
                risk_amount REAL,
                risk_percent REAL,
                target_percent REAL,
                stop_percent REAL,
                analytical_text TEXT,
                client_name TEXT,
                status TEXT DEFAULT 'ABERTA',
                pdf_path TEXT,
                created_at TEXT NOT NULL
            )
            """
            )
            
            # Day Trade Sessions table
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS day_trade_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL,
                timeframe_major TEXT DEFAULT '1h',
                timeframe_minor TEXT DEFAULT '15m',
                risk_amount REAL,
                risk_percent REAL,
                pdf_path TEXT,
                created_at TEXT NOT NULL
            )
            """
            )
            
            # Day Trade Entries table
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS day_trade_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry REAL NOT NULL,
                max_entry_variation REAL NOT NULL,
                target REAL NOT NULL,
                stop REAL NOT NULL,
                risk_zero_price REAL,
                risk_zero_percent REAL,
                target_percent REAL,
                stop_percent REAL,
                FOREIGN KEY (session_id) REFERENCES day_trade_sessions(id) ON DELETE CASCADE
            )
            """
            )
            
            # Portfolios table
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_type TEXT NOT NULL,
                state TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                analytical_text TEXT,
                version INTEGER DEFAULT 1,
                pdf_path TEXT,
                created_at TEXT NOT NULL
            )
            """
            )
            
            # Portfolio Assets table
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS portfolio_assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                entry REAL NOT NULL,
                entry_max REAL NOT NULL,
                risk_zero REAL NOT NULL,
                target REAL NOT NULL,
                stop REAL NOT NULL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
            )
            """
            )
            
            con.commit()
            self._ensure_new_columns(cur)
            con.commit()

    def _ensure_new_columns(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("PRAGMA table_info(operations)")
        existing = {row[1] for row in cursor.fetchall()}
        columns_to_add = {
            "entrada_min": "ALTER TABLE operations ADD COLUMN entrada_min REAL",
            "entrada_max": "ALTER TABLE operations ADD COLUMN entrada_max REAL",
            "parcial_preco": "ALTER TABLE operations ADD COLUMN parcial_preco REAL",
            "parcial_pontos": "ALTER TABLE operations ADD COLUMN parcial_pontos REAL",
            "tick_size": "ALTER TABLE operations ADD COLUMN tick_size REAL",
            "risco_retorno": "ALTER TABLE operations ADD COLUMN risco_retorno REAL",
        }

        for column, ddl in columns_to_add.items():
            if column not in existing:
                cursor.execute(ddl)

    def insert_operation(
        self,
        operation: Operation,
        pdf_path: Optional[str] = None,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO operations (
                    symbol, tipo, entrada, entrada_min, entrada_max,
                    stop, alvo, parcial_preco, parcial_pontos, quantidade,
                    observacoes, preco_atual, pontos_alvo, pontos_stop,
                    tick_size, risco_retorno, status, created_at,
                    pdf_path, timeframe, indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    operation.symbol,
                    operation.tipo,
                    operation.entrada,
                    operation.entrada_min,
                    operation.entrada_max,
                    operation.stop,
                    operation.alvo,
                    operation.parcial_preco,
                    operation.parcial_pontos,
                    operation.quantidade,
                    operation.observacoes,
                    operation.preco_atual,
                    operation.pontos_alvo,
                    operation.pontos_stop,
                    operation.tick_size,
                    operation.risco_retorno,
                    operation.status,
                    operation.created_at,
                    pdf_path,
                    operation.timeframe,
                    json.dumps(indicators) if indicators else "{}",
                ),
            )
            con.commit()
            return cur.lastrowid

    def get_operations(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT id, symbol, tipo, entrada, entrada_min, entrada_max,
                       stop, alvo, parcial_preco, parcial_pontos,
                       quantidade, observacoes, preco_atual, pontos_alvo,
                       pontos_stop, tick_size, risco_retorno,
                       status, created_at, pdf_path, timeframe
                FROM operations
                ORDER BY id DESC
                LIMIT ?
            """,
                (limit,),
            )
            rows = cur.fetchall()

            operations: List[Dict[str, Any]] = []
            for row in rows:
                operations.append(
                    {
                        "id": row[0],
                        "symbol": row[1],
                        "tipo": row[2],
                        "entrada": row[3],
                        "entrada_min": row[4],
                        "entrada_max": row[5],
                        "stop": row[6],
                        "alvo": row[7],
                        "parcial_preco": row[8],
                        "parcial_pontos": row[9],
                        "quantidade": row[10],
                        "observacoes": row[11],
                        "preco_atual": row[12],
                        "pontos_alvo": row[13],
                        "pontos_stop": row[14],
                        "tick_size": row[15],
                        "risco_retorno": row[16],
                        "status": row[17],
                        "created_at": row[18],
                        "pdf_path": row[19],
                        "timeframe": row[20],
                    }
                )

            return operations

    # ========== SWING TRADE METHODS ==========
    def insert_swing_trade(self, swing: Dict[str, Any]) -> int:
        """Insert a swing trade operation."""
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO swing_trades (
                    symbol, direction, entry, entry_min, entry_max,
                    target, stop, quantity, trade_date,
                    timeframe_major, timeframe_minor,
                    risk_amount, risk_percent, target_percent, stop_percent,
                    analytical_text, client_name, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    swing["symbol"], swing["direction"], swing["entry"],
                    swing["entry_min"], swing["entry_max"],
                    swing["target"], swing["stop"], swing["quantity"],
                    swing["trade_date"], swing.get("timeframe_major", "1d"),
                    swing.get("timeframe_minor", "1h"),
                    swing.get("risk_amount"), swing.get("risk_percent"),
                    swing.get("target_percent"), swing.get("stop_percent"),
                    swing.get("analytical_text", ""),
                    swing.get("client_name"),
                    swing.get("status", "ABERTA"),
                    swing.get("created_at")
                )
            )
            con.commit()
            return cur.lastrowid

    def update_swing_trade_pdf(self, trade_id: int, pdf_path: str) -> None:
        """Update PDF path for a swing trade."""
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("UPDATE swing_trades SET pdf_path = ? WHERE id = ?", (pdf_path, trade_id))
            con.commit()

    def get_swing_trades(self, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """Get swing trades.

        When limit is None, returns all rows (no LIMIT). This avoids SQLite 'datatype mismatch'
        errors caused by binding NULL to a LIMIT parameter.
        """
        with self._connect() as con:
            cur = con.cursor()
            base_query = """
                SELECT id, symbol, direction, entry, entry_min, entry_max,
                       target, stop, quantity, trade_date,
                       timeframe_major, timeframe_minor,
                       risk_amount, risk_percent, target_percent, stop_percent,
                       analytical_text, client_name, status, pdf_path, created_at
                FROM swing_trades
                ORDER BY id DESC
            """

            if limit is None:
                cur.execute(base_query)
            else:
                cur.execute(base_query + " LIMIT ?", (int(limit),))

            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "symbol": r[1],
                "direction": r[2],
                "entry": r[3],
                "entry_min": r[4],
                "entry_max": r[5],
                "target": r[6],
                "stop": r[7],
                "quantity": r[8],
                "trade_date": r[9],
                "timeframe_major": r[10],
                "timeframe_minor": r[11],
                "risk_amount": r[12],
                "risk_percent": r[13],
                "target_percent": r[14],
                "stop_percent": r[15],
                "analytical_text": r[16],
                "client_name": r[17],
                "status": r[18],
                "pdf_path": r[19],
                "created_at": r[20],
            }
            for r in rows
        ]

    def get_swing_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        trades = self.get_swing_trades(limit=None)
        return next((t for t in trades if t.get("id") == trade_id), None)

    # ========== DAY TRADE METHODS ==========
    def insert_day_trade_session(self, session: Dict[str, Any], entries: List[Dict[str, Any]]) -> int:
        """Insert a day trade session with entries."""
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO day_trade_sessions (
                    trade_date, timeframe_major, timeframe_minor,
                    risk_amount, risk_percent, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session["trade_date"],
                    session.get("timeframe_major", "1h"),
                    session.get("timeframe_minor", "15m"),
                    session.get("risk_amount"),
                    session.get("risk_percent"),
                    session.get("created_at")
                )
            )
            session_id = cur.lastrowid
            
            for entry in entries:
                cur.execute(
                    """
                    INSERT INTO day_trade_entries (
                        session_id, symbol, direction, entry, max_entry_variation,
                        target, stop, risk_zero_price, risk_zero_percent,
                        target_percent, stop_percent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id, entry["symbol"], entry["direction"],
                        entry["entry"], entry["max_entry_variation"],
                        entry["target"], entry["stop"],
                        entry.get("risk_zero_price"), entry.get("risk_zero_percent"),
                        entry.get("target_percent"), entry.get("stop_percent")
                    )
                )
            
            con.commit()
            return session_id

    def update_day_trade_session_pdf(self, session_id: int, pdf_path: str) -> None:
        """Update PDF path for a day trade session."""
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("UPDATE day_trade_sessions SET pdf_path = ? WHERE id = ?", (pdf_path, session_id))
            con.commit()

    def get_day_trade_sessions(self, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """Get day trade sessions with entries."""
        with self._connect() as con:
            cur = con.cursor()
            base_query = """
                SELECT id, trade_date, timeframe_major, timeframe_minor,
                       risk_amount, risk_percent, pdf_path, created_at
                FROM day_trade_sessions
                ORDER BY id DESC
            """
            if limit is None:
                cur.execute(base_query)
            else:
                cur.execute(base_query + " LIMIT ?", (int(limit),))
            sessions = []
            for row in cur.fetchall():
                session_id = row[0]
                cur.execute(
                    """
                    SELECT symbol, direction, entry, max_entry_variation,
                           target, stop, risk_zero_price, risk_zero_percent,
                           target_percent, stop_percent
                    FROM day_trade_entries
                    WHERE session_id = ?
                    """,
                    (session_id,)
                )
                entries = [
                    {
                        "symbol": e[0], "direction": e[1], "entry": e[2],
                        "max_entry_variation": e[3], "target": e[4], "stop": e[5],
                        "risk_zero_price": e[6], "risk_zero_percent": e[7],
                        "target_percent": e[8], "stop_percent": e[9]
                    }
                    for e in cur.fetchall()
                ]
                sessions.append({
                    "id": session_id, "trade_date": row[1],
                    "timeframe_major": row[2], "timeframe_minor": row[3],
                    "risk_amount": row[4], "risk_percent": row[5],
                    "pdf_path": row[6], "created_at": row[7],
                    "entries": entries
                })
            return sessions

    def get_day_trade_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        sessions = self.get_day_trade_sessions(limit=None)
        return next((s for s in sessions if s.get("id") == session_id), None)

    def get_manipulated_assets(
        self,
        start_date: str,
        end_date: str,
        include_swing: bool = True,
        include_daytrade: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return a de-duplicated list of assets manipulated in Swing/Day Trade.

        The result is grouped by (source, symbol) and keeps the most recent record in the
        requested date range.

        Dates are expected in ISO format: YYYY-MM-DD.
        """

        results: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        with self._connect() as con:
            cur = con.cursor()

            if include_swing:
                cur.execute(
                    """
                    SELECT symbol, entry, entry_max, target, stop, trade_date, id
                    FROM swing_trades
                    WHERE trade_date >= ? AND trade_date <= ?
                    ORDER BY trade_date DESC, id DESC
                    """,
                    (start_date, end_date),
                )
                for symbol, entry, entry_max, target, stop, trade_date, _trade_id in cur.fetchall():
                    sym = str(symbol or "").upper()
                    key = ("swing", sym)
                    if not sym or key in seen:
                        continue
                    seen.add(key)
                    results.append(
                        {
                            "source": "swing",
                            "symbol": sym,
                            "entry": entry,
                            "entry_max": entry_max,
                            "risk_zero": entry,
                            "target": target,
                            "stop": stop,
                            "trade_date": trade_date,
                        }
                    )

            if include_daytrade:
                cur.execute(
                    """
                    SELECT e.symbol, e.entry, e.max_entry_variation, e.target, e.stop, e.risk_zero_price,
                           s.trade_date, s.id, e.id
                    FROM day_trade_entries e
                    JOIN day_trade_sessions s ON s.id = e.session_id
                    WHERE s.trade_date >= ? AND s.trade_date <= ?
                    ORDER BY s.trade_date DESC, s.id DESC, e.id DESC
                    """,
                    (start_date, end_date),
                )
                for (
                    symbol,
                    entry,
                    max_entry_variation,
                    target,
                    stop,
                    risk_zero_price,
                    trade_date,
                    _session_id,
                    _entry_id,
                ) in cur.fetchall():
                    sym = str(symbol or "").upper()
                    key = ("daytrade", sym)
                    if not sym or key in seen:
                        continue
                    seen.add(key)

                    computed_entry_max = entry
                    try:
                        entry_f = float(entry)
                        var_f = float(max_entry_variation or 0)
                        computed_entry_max = entry_f * (1.0 + (var_f / 100.0))
                    except Exception:
                        computed_entry_max = entry

                    risk_zero = risk_zero_price if risk_zero_price is not None else entry

                    results.append(
                        {
                            "source": "daytrade",
                            "symbol": sym,
                            "entry": entry,
                            "entry_max": computed_entry_max,
                            "risk_zero": risk_zero,
                            "target": target,
                            "stop": stop,
                            "trade_date": trade_date,
                        }
                    )

        return results

    # ========== PORTFOLIO METHODS ==========
    def insert_portfolio(self, portfolio: Dict[str, Any], assets: List[Dict[str, Any]]) -> int:
        """Insert a portfolio with assets."""
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO portfolios (
                    portfolio_type, state, start_date, end_date,
                    analytical_text, version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    portfolio["portfolio_type"], portfolio["state"],
                    portfolio.get("start_date"), portfolio.get("end_date"),
                    portfolio.get("analytical_text", ""),
                    portfolio.get("version", 1),
                    portfolio.get("created_at")
                )
            )
            portfolio_id = cur.lastrowid
            
            for asset in assets:
                cur.execute(
                    """
                    INSERT INTO portfolio_assets (
                        portfolio_id, symbol, entry, entry_max,
                        risk_zero, target, stop
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        portfolio_id, asset["symbol"], asset["entry"],
                        asset["entry_max"], asset["risk_zero"],
                        asset["target"], asset["stop"]
                    )
                )
            
            con.commit()
            return portfolio_id

    def update_portfolio_pdf(self, portfolio_id: int, pdf_path: str) -> None:
        """Update PDF path for a portfolio."""
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("UPDATE portfolios SET pdf_path = ? WHERE id = ?", (pdf_path, portfolio_id))
            con.commit()

    def get_portfolios(self, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """Get portfolios with assets."""
        with self._connect() as con:
            cur = con.cursor()
            base_query = """
                SELECT id, portfolio_type, state, start_date, end_date,
                       analytical_text, version, pdf_path, created_at
                FROM portfolios
                ORDER BY id DESC
            """
            if limit is None:
                cur.execute(base_query)
            else:
                cur.execute(base_query + " LIMIT ?", (int(limit),))
            portfolios = []
            for row in cur.fetchall():
                portfolio_id = row[0]
                cur.execute(
                    """
                    SELECT symbol, entry, entry_max, risk_zero, target, stop
                    FROM portfolio_assets
                    WHERE portfolio_id = ?
                    """,
                    (portfolio_id,)
                )
                assets = [
                    {
                        "symbol": a[0], "entry": a[1], "entry_max": a[2],
                        "risk_zero": a[3], "target": a[4], "stop": a[5]
                    }
                    for a in cur.fetchall()
                ]
                portfolios.append({
                    "id": portfolio_id, "portfolio_type": row[1],
                    "state": row[2], "start_date": row[3], "end_date": row[4],
                    "analytical_text": row[5], "version": row[6],
                    "pdf_path": row[7], "created_at": row[8],
                    "assets": assets
                })
            return portfolios

    def get_portfolio(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        portfolios = self.get_portfolios(limit=None)
        return next((p for p in portfolios if p.get("id") == portfolio_id), None)
