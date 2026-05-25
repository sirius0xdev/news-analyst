import psycopg2
import os
import logging
from google import genai  # Use the new 2026 SDK
import yfinance as yf
from datetime import datetime
import time
import requests 
# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "postgres-service").strip(),
    "database": os.getenv("DB_NAME", "news_app_db").strip(),
    "user": os.getenv("DB_USER", "news_app").strip(),
    "password": os.getenv("DB_PASSWORD", "").strip(),
    "port": os.getenv("DB_PORT", "5432").strip()
}

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
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')  # or EDT if preferred
    
    for category, tickers in FUTURES_TICKERS.items():
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="1d", interval="1m")
                if not data.empty:
                    last_price = data['Close'].iloc[-1]
                    prices[ticker] = {
                        'price': round(last_price, 2),
                        'change_pct': round((last_price - data['Open'].iloc[0]) / data['Open'].iloc[0] * 100, 2) if len(data) > 1 else 0,
                        'timestamp': timestamp,
                        'category': category
                    }
                else:
                    prices[ticker] = {'price': None, 'error': 'No data'}
            except Exception as e:
                prices[ticker] = {'price': None, 'error': str(e)}
    
    return prices
def get_recent_news():
    """Fetch news from the last 1 hour."""
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
# 1. Initialize the Gemini Client
# It automatically looks for the GEMINI_API_KEY environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.0-flash"  # Highly efficient for summarization tasks

def call_llm(prompt):
    """Sends a prompt to the Gemini API."""
    try:
        # 2. Update the call logic to use the SDK instead of requests
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        print(f"Raw response type: {type(response)}")
        print(f"Full response object: {response}")
        return response.response 
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return ""

# ... rest of your get_recent_news() and save_summary_to_db() logic remains the same ...

def summarize_news():
    articles = get_recent_news()
    if not articles:
        logging.info("No new articles found.")
        return

    logging.info(f"Processing {len(articles)} articles with Gemini...")
    
    current_prices = fetch_current_futures_prices()
    
    # Build clean, readable context block for the LLM
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
    price_context = f"CURRENT FUTURES PRICES (as of {timestamp_str}):\n"
    
    for ticker, info in current_prices.items():
        if info.get('price') is not None:
            price_context += (
                f"- {ticker} ({info['category']}): ${info['price']:.2f}  "
                f"({info['change_pct']:+.2f}% today)\n"
            )
        else:
            price_context += f"- {ticker}: unavailable ({info.get('error', 'unknown error')})\n"
    
    price_context += (
        "\nUse these **exact** current levels as context for any market impact assessment, "
        "trading edges, entry/exit ideas or directional bias. "
        "Reference them explicitly in your output when relevant "
        "(example: 'With CL currently at $78.12, this news increases bearish risk below current price').\n"
    )
    partial_summaries = []
    for i in range(0, len(articles), 50):
        batch = articles[i:i+50]
        batch_text = "\n\n".join([f"Title: {a[0]}\nContent: {a[1][:1500]}" for a in batch])
        
        
        map_prompt_default = f"""
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
    FINAL_PROMPT_DEFAULT = f"""
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
        query = """"CREATE TABLE IF NOT EXISTS article_summaries (
                id SERIAL PRIMARY KEY,
                summary_text TEXT NOT NULL,
                is_master_summary BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO news_app;
                GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO news_app;
                INSERT INTO article_summaries (summary_text) VALUES (%s);"""
        cur.execute(query, (summary_text,))
        
        conn.commit()
        print("Master summary saved to database successfully.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error saving summary to DB: {e}")

if __name__ == "__main__":
    summarize_news() 

