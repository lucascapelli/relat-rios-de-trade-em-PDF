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

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
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
        with sqlite3.connect(self.db_path) as con:
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
        with sqlite3.connect(self.db_path) as con:
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
