import os
import json
import csv
import re
from datetime import datetime, timezone
import requests
from pydantic import BaseModel, Field
from typing import Literal
from google import genai
from google.genai import types

STATE_FILE = "portfolio_state.json"
HISTORY_FILE = "portfolio_history.csv"
TRADES_FILE = "trades_log.csv"

class TradingDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="Handelsaktion: BUY, SELL oder HOLD."
    )
    symbol: Literal["BTC", "ETH", "SOL"] = Field(
        description="Das Krypto-Symbol."
    )
    allocation_pct: float = Field(
        description="Anteil des verfügbaren Cashs (bei BUY) oder des Bestands (bei SELL) zwischen 0.05 und 1.0. Bei HOLD: 0.0."
    )
    reasoning: str = Field(
        description="Konkrete, faktenbasierte Begründung basierend auf aktuellen News und Marktlage."
    )

def get_live_prices():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd"
    try:
        res = requests.get(url, timeout=10).json()
        return {
            "BTC": float(res["bitcoin"]["usd"]),
            "ETH": float(res["ethereum"]["usd"]),
            "SOL": float(res["solana"]["usd"]),
        }
    except Exception as e:
        print(f"Fehler beim Preisabruf: {e}")
        return None

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"cash": 10000.0, "holdings": {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def ask_gemini(prices, state):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    prompt = f"""
    Du bist ein disziplinierter quantitativer Krypto-Trading-Bot.
    
    Aktuelle Kurse (USD):
    BTC: ${prices['BTC']:,.2f}
    ETH: ${prices['ETH']:,.2f}
    SOL: ${prices['SOL']:,.2f}
    
    Portfolio-Status:
    Verfügbares Cash: ${state['cash']:,.2f}
    Bestände: {state['holdings']}
    
    Aufgabe:
    1. Nutze Google Search für die aktuellen Krypto-Schlagzeilen und Markttrends der letzten 24 Stunden.
    2. Analysiere Sentiment, Momentum und Risiko.
    3. Triff eine disziplinierte Entscheidung (BUY, SELL oder HOLD). Riskiere bei Neukäufen nie mehr als 25% des Cashs pro Trade (allocation_pct <= 0.25).
    
    WICHTIG: Antworte AUSSCHLIESSLICH im folgenden JSON-Format ohne weiteren Fließtext davor oder danach (kein Markdown-Block, keine Backticks):
    {{
      "action": "BUY" | "SELL" | "HOLD",
      "symbol": "BTC" | "ETH" | "SOL",
      "allocation_pct": 0.0,
      "reasoning": "Kurze Begründung auf Basis der aktuellen Recherche"
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            temperature=0.2
        )
    )
    
    # JSON-Objekt sicher aus dem Antworttext extrahieren
    text = response.text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = text

    return TradingDecision.model_validate_json(json_str)

def run_cycle():
    prices = get_live_prices()
    if not prices:
        print("Abbruch: Keine Live-Preise erhalten.")
        return

    state = load_state()
    decision = ask_gemini(prices, state)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    trade_row = {
        "timestamp": now_iso,
        "action": decision.action,
        "symbol": decision.symbol,
        "price": prices.get(decision.symbol, 0.0),
        "amount_usd": 0.0,
        "units": 0.0,
        "reasoning": decision.reasoning
    }

    if decision.action == "BUY" and decision.symbol in state["holdings"]:
        invest_cash = state["cash"] * min(max(decision.allocation_pct, 0.0), 0.25)
        if invest_cash >= 15.0:
            units = invest_cash / prices[decision.symbol]
            state["cash"] -= invest_cash
            state["holdings"][decision.symbol] += units
            trade_row["amount_usd"] = round(invest_cash, 2)
            trade_row["units"] = round(units, 6)

    elif decision.action == "SELL" and decision.symbol in state["holdings"]:
        avail_units = state["holdings"][decision.symbol]
        sell_units = avail_units * min(max(decision.allocation_pct, 0.0), 1.0)
        sell_value = sell_units * prices[decision.symbol]
        if sell_value >= 15.0:
            state["cash"] += sell_value
            state["holdings"][decision.symbol] -= sell_units
            trade_row["amount_usd"] = round(sell_value, 2)
            trade_row["units"] = round(sell_units, 6)

    save_state(state)

    # 1. Trade-Log schreiben (auch HOLD-Entscheidungen mit Begründung festhalten)
    file_exists = os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trade_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_row)

    # 2. Portfolio-Snapshot für das Verlaufsdiagramm speichern
    crypto_val = sum(state["holdings"][s] * prices[s] for s in state["holdings"])
    total_val = state["cash"] + crypto_val

    hist_row = {
        "timestamp": now_iso,
        "cash_usd": round(state["cash"], 2),
        "crypto_usd": round(crypto_val, 2),
        "total_usd": round(total_val, 2)
    }
    hist_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hist_row.keys()))
        if not hist_exists:
            writer.writeheader()
        writer.writerow(hist_row)

    print(f"[{now_iso}] {decision.action} {decision.symbol} | Total: ${total_val:,.2f} | Info: {decision.reasoning}")

if __name__ == "__main__":
    run_cycle()
