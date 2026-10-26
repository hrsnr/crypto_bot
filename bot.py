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
    reasoning: str

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

    @field_validator("reasoning", mode="before")
    @classmethod
    def sanitize_reasoning(cls, v):
        if not v or not str(v).strip():
            return "Marktlage neutral/abwartend bewertet – keine Trendimpulse identifiziert."
        return str(v).strip()


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
    """Lädt den Portfoliostatus oder initialisiert ihn; migriert entry_prices bei Bedarf."""
    state = {
        "cash": INITIAL_CASH,
        "holdings": {symbol: 0.0 for symbol in SUPPORTED_COINS.values()},
        "entry_prices": {symbol: 0.0 for symbol in SUPPORTED_COINS.values()},
    }

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                state["cash"] = float(loaded.get("cash", INITIAL_CASH))
                state["holdings"] = {
                    sym: float(loaded.get("holdings", {}).get(sym, 0.0))
                    for sym in SUPPORTED_COINS.values()
                }
                # Migration: Falls entry_prices bisher nicht existierte
                loaded_entry = loaded.get("entry_prices", {})
                state["entry_prices"] = {
                    sym: float(loaded_entry.get(sym, 0.0))
                    for sym in SUPPORTED_COINS.values()
                }
        except Exception as e:
            print(f"[WARNUNG] Konnte {STATE_FILE} nicht lesen: {e}")

    return state


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

    # Detaillierte Positions-Übersicht inklusive P&L zusammenbauen
    positions_summary = []
    total_crypto_val = 0.0
    for sym in SUPPORTED_COINS.values():
        units = state["holdings"].get(sym, 0.0)
        curr_price = prices.get(sym, 0.0)
        entry_price = state["entry_prices"].get(sym, 0.0)
        curr_val = units * curr_price
        total_crypto_val += curr_val

        if units > 0.00001:
            if entry_price > 0:
                pnl_pct = ((curr_price - entry_price) / entry_price) * 100
                pnl_usd = (curr_price - entry_price) * units
                pnl_str = f"P&L: {pnl_pct:+.2f}% ({pnl_usd:+.2f} USD)"
            else:
                pnl_str = "Einstiegskurs: unbekannt"
            positions_summary.append(
                f"- {sym}: {units:.4f} Stk (Wert: ${curr_val:,.2f} | Einstieg: ${entry_price:,.2f} | Kurs: ${curr_price:,.2f} | {pnl_str})"
            )
        else:
            positions_summary.append(
                f"- {sym}: Kein Bestand (Aktueller Kurs: ${curr_price:,.2f})"
            )

    total_portfolio = state["cash"] + total_crypto_val
    cash_ratio = (state["cash"] / total_portfolio) * 100 if total_portfolio > 0 else 100.0

    prompt = f"""
Du bist ein professioneller, taktisch agierender Krypto-Trading-Bot für ein Paper-Trading-Portfolio.
Aktuelle UTC-Zeit: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

PORTFOLIO-ÜBERSICHT:
- Gesamtwert: ${total_portfolio:,.2f} USD
- Freies Cash: ${state['cash']:,.2f} USD ({cash_ratio:.1f}% Cash-Quote)
- Krypto-Bestandswert: ${total_crypto_val:,.2f} USD

BESTÄNDE & POSITIONEN (inkl. P&L):
{chr(10).join(positions_summary)}

STRATEGIE & HANDELSRICHTLINIEN:
1. AKTIVE CHANCENNUTZUNG (DIP-BUYING):
   - Wenn deine Cash-Quote hoch ist (> 50 %), lasse das Geld nicht untätig liegen. Nutze kurzfristige Dips, Bodenbildungen oder positive Momentum-News bei BTC, ETH oder SOL für einen Einstieg.
   - Maximal 25% des freien Cashs (${state['cash'] * 0.25:,.2f} USD) pro Kauforder einsetzen.

2. GEWINNMITNAHME (TAKE-PROFIT):
   - Steht eine gehaltene Position deutlich im Plus (z. B. > +3% bis +8%) oder zeigt der Markt Ermüdungserscheinungen an Widerständen, realisiere Teilgewinne (SELL von z. B. 30% bis 70% des Bestands).

3. RISIKOMANAGEMENT (STOP-LOSS):
   - Wenn eine Position stark ins Minus rutscht (z. B. > -5%) und die Nachrichtenlage weiteren Verkaufsdruck signalisiert, reduziere oder schließe die Position konsequent.

4. DISZIPLINIERTES HOLD:
   - Wähle "HOLD" nur, wenn der Markt exakt konsolidiert oder kein klares Setup vorliegt.
   - PFLICHT: Auch bei "HOLD" musst du in 1-3 Sätzen konkret begründen, warum du abwartest (unter Nennung von Kursniveaus, News-Recherche oder deiner P&L).

AUFGABE:
Nutze Google Search für BTC, ETH und SOL der letzten 24 Stunden (Makro-News, ETF-Zuflüsse, Liquidationsdaten, Sentiments). Triff dann genau EINE Entscheidung (BUY, SELL oder HOLD).
Antworte AUSSCHLIESSLICH im folgenden JSON-Format:

{{
  "action": "BUY" | "SELL" | "HOLD",
  "symbol": "BTC" | "ETH" | "SOL" | null,
  "amount_usd": <Betrag in USD als Zahl; bei HOLD 0.0>,
  "reasoning": "<1-3 Sätze konkrete Analyse mit Bezug auf Recherche/P&L, zwingend erforderlich>"
}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.35,  # Etwas dynamischer für aktive Marktbeurteilung
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

    # Initialer Fix für Altdaten: Falls Bestände vorhanden sind, aber kein Einstiegskurs bekannt ist
    for sym, units in state["holdings"].items():
        if units > 0.00001 and state["entry_prices"].get(sym, 0.0) <= 0.0:
            state["entry_prices"][sym] = prices.get(sym, 0.0)

    decision = ask_gemini(prices, state)
    print(f"Entscheidung: {decision.action} | Symbol: {decision.symbol} | Betrag: ${decision.amount_usd:,.2f}")
    print(f"Begründung: {decision.reasoning}")

    action = decision.action
    symbol = decision.symbol
    requested_amount = max(0.0, float(decision.amount_usd))
    executed_amount = 0.0
    trade_price = prices.get(symbol, 0.0) if symbol else 0.0

    # 1. Trade-Logik anwenden & Einstiegskurse nachführen
    if action == "BUY" and symbol in prices and prices[symbol] > 0:
        max_allowed_cash = state["cash"] * 0.25
        trade_usd = min(requested_amount, max_allowed_cash, state["cash"])
        if trade_usd >= 10.0:
            units_bought = trade_usd / prices[symbol]
            old_units = state["holdings"].get(symbol, 0.0)
            old_entry = state["entry_prices"].get(symbol, 0.0)

            # Neuer gewichteter Durchschnitts-Einstiegspreis
            new_total_units = old_units + units_bought
            new_avg_entry = (
                ((old_units * old_entry) + trade_usd) / new_total_units
                if new_total_units > 0
                else prices[symbol]
            )

            state["cash"] -= trade_usd
            state["holdings"][symbol] = new_total_units
            state["entry_prices"][symbol] = new_avg_entry
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
            remaining_units = max(0.0, current_units - units_to_sell)
            state["holdings"][symbol] = remaining_units
            state["cash"] += trade_usd
            executed_amount = trade_usd

            # Wenn Position komplett aufgelöst wurde, Einstiegspreis zurücksetzen
            if remaining_units <= 0.00001:
                state["holdings"][symbol] = 0.0
                state["entry_prices"][symbol] = 0.0
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
