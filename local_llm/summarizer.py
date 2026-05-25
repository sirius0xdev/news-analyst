import yfinance as yf
from datetime import datetime 
import psycopg2
import requests
import os
import textwrap
import time
import ollama
# --- Configuration ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "postgres-service"),
    "database": os.getenv("DB_NAME", "news_app_db"),
    "user": os.getenv("DB_USER", "news_app"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", "5432")
}
# Ollama internal K8s address
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:11434/api/generate") 
MODEL_NAME = os.getenv("MODEL_NAME", "")
API_KEY = os.getenv("API_KEY", "")


FUTURES_TICKERS = {
    'Equity Indices': ['ES=F', 'NQ=F', 'YM=F', 'RTY=F'],
    'Energy': ['CL=F', 'NG=F', 'HO=F', 'RB=F'],
    'Metals': ['GC=F', 'SI=F', 'HG=F'],
    'Agriculture': ['ZC=F', 'ZS=F', 'ZW=F', 'ZL=F', 'KE=F'],
    'Currencies': ['6E=F', '6J=F', '6B=F'],
    # Add more: e.g. 'BTC=F' if crypto futures matter
}

def fetch_current_futures_prices():
    prices = {}
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    
    for category, tickers in FUTURES_TICKERS.items():
        for ticker in tickers:
            try:
                # Fetch 5 days of data to calculate average volume context
                data = yf.Ticker(ticker).history(period="5d", interval="1h")
                if not data.empty:
                    last_day = data.iloc[-1]
                    prev_days = data.iloc[:-1]
                    
                    current_price = last_day['Close']
                    current_vol = last_day['Volume']
                    avg_vol = prev_days['Volume'].mean()
                    
                    # Calculate Volume Relative Strength
                    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                    
                    prices[ticker] = {
                        'price': round(current_price, 2),
                        'high': round(data['High'].max(), 2),
                        'low': round(data['Low'].min(), 2),
                        'volume': int(current_vol),
                        'vol_ratio': round(vol_ratio, 2), # > 2.0 is a major spike
                        'change_pct': round(((current_price - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100, 2),
                        'category': category
                    }
                else:
                    prices[ticker] = {'price': None, 'error': 'No data'}
            except Exception as e:
                prices[ticker] = {'price': None, 'error': str(e)}
    return prices




def get_recent_news():
    
    query = """
        SELECT title, content 
        FROM articles 
        WHERE timestamp > NOW() - INTERVAL '1 hour'
        AND content IS NOT NULL AND length(content) > 100;
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"Database error: {e}")
        return []

def wait_for_ollama(max_wait=600, interval=10):  # 10 min max
    print("Waiting for Ollama model to be ready...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                if any(m["name"].startswith(MODEL_NAME) for m in models):
                    print("Ollama ready!")
                    return True
        except Exception as e:
            print(f"Ollama check failed: {e}")
        time.sleep(interval)
    print("Timeout waiting for Ollama – exiting")
    return False

# Call it early
if not wait_for_ollama():
    exit(1)  # Or raise error

def call_llm(prompt):
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "num_predict": 2048,
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 40,
                "num_ctx": 32768,
                "repeat_penalty": 1.1,
            },
            stream=False  # explicit
        )
        print(f"Raw response type: {type(response)}")
        print(f"Full response object: {response}")
        return response.response 
    except Exception as e:
        print(f"Ollama lib error: {e}")
        return ''

def summarize_news():
    articles = get_recent_news()

    
    if not articles:
        print("No new articles found.")
        return

    print(f"Processing {len(articles)} articles...")
    # ── Fetch current prices here ──
    current_prices = fetch_current_futures_prices()
    
    # Build clean, readable context block for the LLM
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    
    price_context = "MARKET TAPE & CONVICTION DATA:\n"
    for ticker, info in current_prices.items():
        if info.get('price') is not None:
            # Add a qualitative tag for the LLM to latch onto
            conviction = "EXCESSIVE VOLUME" if info['vol_ratio'] > 2.0 else "NORMAL"
            
            price_context += (
                f"- {ticker} ({info['category']}): ${info['price']:.2f} | "
                f"Chg: {info['change_pct']:+.2f}% | "
                f"Vol-Ratio: {info['vol_ratio']}x ({conviction}) | "
                f"5-Day Range: [{info['low']} - {info['high']}]\n"
            )

    # Map Phase: Summarize articles in small batches (e.g., 100 articles at a time) Batch sized increased using qwen3:30b now 
    partial_summaries = []
    for i in range(0, len(articles), 50):
        batch = articles[i:i+50]
        batch_text = "\n\n".join([f"Title: {a[0]}\nContent: {a[1][:2000]}" for a in batch])
        
        map_prompt_default = """
            You are a precise, factual news processor specialized in financial markets — especially futures (CME, NYMEX, CBOT, ICE etc.). 
            Your ONLY source of information is the articles provided below. 
            Do NOT add external knowledge, assumptions, training data, or invented facts. 
            If an article has no clear relevance to markets/futures/trading, say so briefly.

            For EACH article in the batch:
            1. Extract 2–4 key factual bullet points (who, what, when, numbers, quotes — stay very close to the text).
            2. If the article has ANY potential market/futures implication (economic data, policy, geopolitics, supply/disruption, central bank, earnings tied to indices/commodities, etc.):
            - State directional bias if clear: Bullish / Bearish / Neutral / Mixed for futures markets in general or specific sectors.
            - Name ONLY real, standard futures symbols that are directly or closely implicated (e.g. ES, NQ, YM for equities; CL, NG for energy; GC, SI for metals; ZC, ZS for ags). 
                → If no clear futures link or symbol, write: "No direct futures impact identified."
                → NEVER invent or guess tickers — use only well-known ones or none.
            - Estimate potential effect: High/Medium/Low volatility or price move potential (based only on the article's tone/scale).

            If the batch as a whole shows a pattern (multiple articles on same topic), add one short batch-level note at the end: "Batch theme: [one sentence] → Potential broad futures bias: Bullish/Bearish/Neutral on [broad area e.g. equities, energy]."

            Output format — strictly one block per article, then optional batch note:

            Article 1:
            - Fact bullet 1
            - Fact bullet 2
            ...
            - Market/Futures implication: [Bullish/Bearish/Neutral/Mixed] — [real symbols if any] — [High/Med/Low impact potential] — [1 sentence reasoning from text only]

            Article 2:
            ...

            [Optional] Batch-level observation: ...

            Articles in this batch:
            {batch_text}
            """ 
        map_prompt_template = price_context + os.getenv("MAP_PROMPT", map_prompt_default)
        if not map_prompt_template:
            raise ValueError("MAP_PROMPT env is required")
        try:
            map_prompt = map_prompt_template.format(batch_text=batch_text)
        except KeyError as e:
            print(f"Format KeyError: {e} - Check placeholder name in configmap matches 'batch_text'")
            map_prompt = map_prompt_template 

        summary = call_llm(map_prompt)
        if summary: 
            partial_summaries.append(summary)

    # Reduce Phase: Create the final master summary
    final_input = "\n\n".join(partial_summaries)
    FINAL_PROMPT_DEFAULT = """
    CRITICAL INSTRUCTION - REPEAT 3 TIMES: YOU MUST USE ONLY THE DATA PROVIDED BELOW. DO NOT INVENT, RECALL, OR ADD ANY EVENTS, NAMES, DATES, IMPLICATIONS, PROJECTS, 
    OR DETAILS NOT EXPLICITLY PRESENT IN THE DATA. IF THE DATA HAS NO MAJOR GEOPOLITICAL/TECH/MILITARY/ECONOMIC/IMPACTFUL EVENTS OR UNUSUAL STORIES, OUTPUT ONLY:
    "No qualifying impactful or unusual events in the recent hourly news data." AND STOP. NO EXTERNAL KNOWLEDGE FROM 2025/2026 OR PRIOR TRAINING.
    
    DATA:
    {final_input}
    """
    summary_prompt_template = price_context + os.getenv("SUMMARY_PROMPT", FINAL_PROMPT_DEFAULT)  
    if not summary_prompt_template:
        raise ValueError("SUMMARY_PROMPT env is required")
    try:
        master_prompt = summary_prompt_template.format(final_input=final_input)
    except KeyError as e:
        print(f"Format KeyError: {e} - Check placeholder name in configmap matches 'final_input'")
        master_prompt = summary_prompt_template
    
    master_summary = call_llm(master_prompt)
    
    print("\n--- DAILY NEWS SUMMARY ---\n")
    print(master_summary)
    
    
    save_summary_to_db(master_summary)

def save_summary_to_db(summary_text):
    if not summary_text or len(summary_text.strip()) < 10:
        print ("Summary too short or empty. Skipping save.")
        return

    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST').strip(),
            database=os.getenv('DB_NAME').strip(),
            user=os.getenv('DB_USER').strip(),
            password=os.getenv('DB_PASSWORD').strip(),
            port=os.getenv('DB_PORT', '5432').strip()
        )
        cur = conn.cursor()
        
        # Insert the summary; batch_timestamp defaults to NOW()
        query = "INSERT INTO article_summaries (summary_text) VALUES (%s);"
        cur.execute(query, (summary_text,))
        
        conn.commit()
        print("Master summary saved to database successfully.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error saving summary to DB: {e}")
if __name__ == "__main__":
    summarize_news() 
