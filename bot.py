import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Literal, Optional

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, field_validator

# --- KONFIGURATION & DATEIPFADE ---
STATE_FILE = "portfolio_state.json"
HISTORY_FILE = "portfolio_history.csv"
TRADES_FILE = "trades_log.csv"

SUPPORTED_COINS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
}
INITIAL_CASH = 10000.0


# --- PYDANTIC MODEL MIT ROBUSTER VALIDIERUNG ---
class TradingDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: Optional[Literal["BTC", "ETH", "SOL"]] = None
    amount_usd: float = 0.0
    reasoning: str = ""

    @field_validator("symbol", mode="before")
    @classmethod
    def sanitize_symbol(cls, v):
        """Wandelt 'None', 'null', 'N/A' oder Leerstrings in echtes Python None um."""
        if v is None:
            return None
        if isinstance(v, str):
            clean = v.strip().upper()
            if clean in ("NONE", "NULL", "N/A", "NAN", "", "–", "-"):
                return None
            return clean
        return v

    @field_validator("amount_usd", mode="before")
    @classmethod
    def sanitize_amount(cls, v):
        """Sichert Fließkomma-Konvertierung gegen None oder fehlerhafte Typen ab."""
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


# --- HELPER-FUNKTIONEN ---
def get_current_prices() -> dict[str, float]:
    """Holt die aktuellen Kurse von CoinGecko mit Fallback-Toleranz."""
    ids = ",".join(SUPPORTED_COINS.keys())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    headers = {"User-Agent": "CryptoTradingBot/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        return {
            SUPPORTED_COINS[coin_id]: float(data[coin_id]["usd"])
            for coin_id in SUPPORTED_COINS
            if coin_id in data
        }
    except Exception as e:
        print(f"[WARNUNG] Kursabfrage fehlgeschlagen: {e}")
        return {}


def load_portfolio_state() -> dict:
    """Lädt den Portfoliostatus oder initialisiert ihn mit 10.000$ Startkapital."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNUNG] Konnte {STATE_FILE} nicht lesen: {e}")

    return {
        "cash": INITIAL_CASH,
        "holdings": {symbol: 0.0 for symbol in SUPPORTED_COINS.values()},
    }


def save_portfolio_state(state: dict):
    """Speichert den aktuellen Kontostand und die Bestände als JSON."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def extract_json_from_response(text: str) -> str:
    """Extrahiert sauberes JSON aus Markdown-Codeblöcken oder Textteilen."""
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    bracket_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if bracket_match:
        return bracket_match.group(1).strip()

    return text.strip()


# --- GEMINI ENTSCHEIDUNGSLOGIK ---
def ask_gemini(prices: dict, state: dict) -> TradingDecision:
    """Recherchiert Marktdaten per Google Search Grounding und entscheidet über Allokation."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[FEHLER] Kein GEMINI_API_KEY vorhanden.")
        return TradingDecision(
            action="HOLD",
            symbol=None,
            amount_usd=0.0,
            reasoning="GEMINI_API_KEY Secret fehlt in GitHub Actions.",
        )

    client = genai.Client(api_key=api_key)

    holdings_str = ", ".join(
        f"{sym}: {qty:.4f}" for sym, qty in state["holdings"].items()
    )
    prices_str = ", ".join(
        f"{sym}: ${val:,.2f}" for sym, val in prices.items()
    )

    prompt = f"""
Du bist ein professioneller, risikobewusster Krypto-Trading-Bot für ein Paper-Trading-Portfolio.
Aktuelle UTC-Zeit: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

PORTFOLIO-STATUS:
- Freies Cash: ${state['cash']:,.2f} USD
- Aktive Bestände: {holdings_str}
- Aktuelle Marktpreise (USD): {prices_str}

AUFGABE:
1. Nutze Google Search, um aktuelle Trends, Marktsentiment und Nachrichten der letzten 24 Stunden zu BTC, ETH und SOL zu prüfen.
2. Entscheide, ob ein BUY, SELL oder HOLD sinnvoll ist.

REGELN:
- HOLD: Wenn der Markt unklar, überhitzt oder volatil ist, wähle diszipliniert "HOLD". In diesem Fall MUSS "symbol": null und "amount_usd": 0.0 sein.
  WICHTIG: Auch bei "HOLD" MUSST du im Feld "reasoning" zwingend in 1-3 Sätzen deine Markteinschätzung begründen (z. B. Chartbild, Marktlage, Makro-News oder fehlende Kauf-/Verkaufsimpulse)!
- BUY: Maximal 25% des verfügbaren Cashs (${state['cash'] * 0.25:,.2f} USD) investieren.
- SELL: Nur Coins verkaufen, die sich bereits im Bestand befinden. Maximaler Gegenwert entspricht dem aktuellen Bestandswert.
- Gib AUSSCHLIESSLICH ein gültiges JSON-Objekt ohne zusätzliche Erklärungen zurück!

JSON-FORMAT:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "symbol": "BTC" | "ETH" | "SOL" | null,
  "amount_usd": <Zahl in USD, bei HOLD immer 0.0>,
  "reasoning": "<Zwingend 1-3 Sätze Analyse zur Marktlage, auch bei HOLD>"
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )

        response_text = response.text or ""
        json_str = extract_json_from_response(response_text)
        return TradingDecision.model_validate_json(json_str)

    except Exception as e:
        print(f"[WARNUNG] Fallback ausgelöst wegen Verarbeitungsfehler: {e}")
        return TradingDecision(
            action="HOLD",
            symbol=None,
            amount_usd=0.0,
            reasoning=f"Automatischer Fallback auf HOLD (Gemini-API/Validierungsfehler: {e})",
        )


# --- CYCLE EXECUTION & CSV RECORDING ---
def run_cycle():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"=== Starte Trading-Zyklus: {now_utc} ===")

    prices = get_current_prices()
    if not prices:
        print("[ABBRUCH] Keine Kursdaten erhalten. Zyklus übersprungen.")
        return

    state = load_portfolio_state()
    decision = ask_gemini(prices, state)
    print(f"Entscheidung: {decision.action} | Symbol: {decision.symbol} | Betrag: ${decision.amount_usd:,.2f}")
    print(f"Begründung: {decision.reasoning}")

    action = decision.action
    symbol = decision.symbol
    requested_amount = max(0.0, float(decision.amount_usd))
    executed_amount = 0.0
    trade_price = prices.get(symbol, 0.0) if symbol else 0.0

    # 1. Trade-Logik anwenden
    if action == "BUY" and symbol in prices and prices[symbol] > 0:
        max_allowed_cash = state["cash"] * 0.25
        trade_usd = min(requested_amount, max_allowed_cash, state["cash"])
        if trade_usd >= 10.0:
            units = trade_usd / prices[symbol]
            state["cash"] -= trade_usd
            state["holdings"][symbol] = state["holdings"].get(symbol, 0.0) + units
            executed_amount = trade_usd
        else:
            action = "HOLD"
            decision.reasoning += " (Trade-Volumen zu gering für Kauforder)"

    elif action == "SELL" and symbol in prices and prices[symbol] > 0:
        current_units = state["holdings"].get(symbol, 0.0)
        max_sell_usd = current_units * prices[symbol]
        trade_usd = min(requested_amount, max_sell_usd)
        if trade_usd >= 5.0 and current_units > 0:
            units_to_sell = trade_usd / prices[symbol]
            state["holdings"][symbol] = max(0.0, current_units - units_to_sell)
            state["cash"] += trade_usd
            executed_amount = trade_usd
        else:
            action = "HOLD"
            decision.reasoning += " (Kein ausreichender Bestand zum Verkauf vorhanden)"

    else:
        action = "HOLD"

    # 2. Portfolio-Gesamtwert berechnen
    crypto_val = sum(
        state["holdings"].get(sym, 0.0) * prices.get(sym, 0.0)
        for sym in SUPPORTED_COINS.values()
    )
    total_val = state["cash"] + crypto_val

    # 3. State speichern
    save_portfolio_state(state)

    # 4. Historie erfassen (portfolio_history.csv) - Exakte Spaltenreihenfolge!
    history_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not history_exists:
            writer.writerow(["timestamp", "cash_usd", "crypto_usd", "total_usd"])
        writer.writerow([
            now_utc,
            f"{state['cash']:.2f}",
            f"{crypto_val:.2f}",
            f"{total_val:.2f}",
        ])

    # 5. Transaktions-Log schreiben (trades_log.csv)
    reasoning_text = (decision.reasoning or "").replace("\n", " ").strip()
    if not reasoning_text:
        reasoning_text = f"Marktlage abwartend bewertet ({action}) – keine signifikanten Trendimpulse gefunden."

    trades_exists = os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not trades_exists:
            writer.writerow(["timestamp", "action", "symbol", "amount_usd", "price", "reasoning"])
        writer.writerow([
            now_utc,
            action,
            symbol if (action != "HOLD" and symbol) else "–",
            f"{executed_amount:.2f}",
            f"{trade_price:.2f}" if trade_price > 0 else "–",
            reasoning_text,
        ])

    print(f"Zyklus abgeschlossen. Portfolio-Gesamtwert: ${total_val:,.2f}")


if __name__ == "__main__":
    run_cycle()
