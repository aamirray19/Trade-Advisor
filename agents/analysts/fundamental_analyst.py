from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests
import os
import json
import datetime as dt


# ---------------------------
# Alpha Vantage fetch helpers
# ---------------------------

class AlphaVantageClient:
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, function: str, symbol: str):
        url = f"{self.BASE}?function={function}&symbol={symbol}&apikey={self.api_key}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()

    def income_statement(self, symbol: str):
        return self._get("INCOME_STATEMENT", symbol)

    def balance_sheet(self, symbol: str):
        return self._get("BALANCE_SHEET", symbol)

    def cash_flow(self, symbol: str):
        return self._get("CASH_FLOW", symbol)

    def insider_transactions(self, symbol: str):
        return self._get("INSIDER_TRANSACTIONS", symbol)

    def insider_sentiment(self, symbol: str):
        return self._get("INSIDER_SENTIMENT", symbol)


# ---------------------------
# Node definition
# ---------------------------

def create_fundamentals_analyst(llm):
    """
    Fundamental Analyst Node:
    - Pulls Alpha Vantage data (last week)
    - Creates JSON snapshot
    - Uses LLM to generate a Markdown report
    """

    def fundamentals_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]

        av = AlphaVantageClient(os.getenv("ALPHAVANTAGE_API_KEY"))

        # Fetch data
        income = av.income_statement(ticker)
        balance = av.balance_sheet(ticker)
        cash = av.cash_flow(ticker)
        insider_txn = av.insider_transactions(ticker)
        insider_sent = av.insider_sentiment(ticker)

        # Filter last week only
        one_week_ago = (dt.datetime.strptime(current_date, "%Y-%m-%d") - dt.timedelta(days=7)).strftime("%Y-%m-%d")

        def filter_last_week(data, key="fiscalDateEnding"):
            reports = data.get("annualReports", []) + data.get("quarterlyReports", [])
            return [r for r in reports if r.get(key, "") >= one_week_ago]

        fundamentals_snapshot = {
            "symbol": ticker,
            "date": current_date,
            "income_statement": filter_last_week(income),
            "balance_sheet": filter_last_week(balance),
            "cash_flow": filter_last_week(cash),
            "insider_transactions": [
                txn for txn in insider_txn.get("transactions", [])
                if txn.get("transactionDate", "") >= one_week_ago
            ],
            "insider_sentiment": [
                sent for sent in insider_sent.get("data", [])
                if sent.get("month", "") >= one_week_ago[:7]  # monthly data
            ],
        }

        # ---------------------------
        # LLM Prompting
        # ---------------------------

        system_message = (
            "You are a professional equity research analyst. "
            "Analyze the provided company's fundamentals for the past week. "
            "Use income statement, balance sheet, cash flow, insider transactions, "
            "and insider sentiment. Create a structured Markdown report:\n"
            "1. **Company Overview**\n"
            "2. **Financial Highlights** (revenues, profits, margins)\n"
            "3. **Cash Flow & Balance Sheet**\n"
            "4. **Insider Activity** (sentiment & transactions)\n"
            "5. **Risks & Red Flags**\n"
            "6. **Investment View** — Bullish/Neutral/Bearish with 3 drivers\n\n"
            "End with a summary Markdown table of key points."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_message}\n\nCurrent date: {current_date}\nCompany: {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(
            system_message=system_message,
            current_date=current_date,
            ticker=ticker,
        )

        chain = prompt | llm

        result = chain.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Here is the JSON data for {ticker}:\n```json\n{json.dumps(fundamentals_snapshot, indent=2)}\n```",
                    }
                ]
            }
        )

        return {
            "messages": [result],
            "fundamentals_report": result.content,
        }

    return fundamentals_analyst_node
