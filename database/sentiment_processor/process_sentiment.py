import os
import time
import requests
import psycopg2
from psycopg2 import sql
from datetime import datetime

# Configuration
SENTIMENT_API_URL = os.getenv('SENTIMENT_API_URL')
DATABASE_URL = os.getenv('DATABASE_URL')
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 2


def connect_db():
    return psycopg2.connect(DATABASE_URL)


def analyze_sentiment(text):
    """Get sentiment analysis from API"""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                SENTIMENT_API_URL,
                json={'text': text},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"Sentiment API error (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return {'sentiment': 'unknown', 'confidence': 0.0}

'''
def process_batch(conn):
    """Process a batch of comments without sentiment analysis"""
    with conn.cursor() as cur:
        # Get unprocessed comments
        cur.execute("""
        SELECT comment_id, text FROM comments 
        WHERE sentiment_processed = FALSE
        LIMIT %s
        FOR UPDATE SKIP LOCKED
        """, (BATCH_SIZE,))

        batch = cur.fetchall()
        if not batch:
            return False

        # Process each comment
        for comment_id, text in batch:
            try:
                analysis = analyze_sentiment(text)

                # Insert sentiment analysis
                cur.execute("""
                INSERT INTO sentiment_analysis (comment_id, sentiment, confidence)
                VALUES (%s, %s, %s)
                ON CONFLICT (comment_id) DO UPDATE
                SET sentiment = EXCLUDED.sentiment,
                    confidence = EXCLUDED.confidence,
                    processed_at = NOW()
                """, (comment_id, analysis['sentiment'], analysis['confidence']))

                # Mark as processed
                cur.execute("""
                UPDATE comments 
                SET sentiment_processed = TRUE 
                WHERE comment_id = %s
                """, (comment_id,))

                conn.commit()
            except Exception as e:
                print(f"Error processing sentiment for comment {comment_id}: {str(e)}")
                conn.rollback()

        return True
'''


def process_batch(conn):
    """Process a batch of comments without sentiment analysis"""
    with conn.cursor() as cur:
        # Get unprocessed comments
        cur.execute("""
        SELECT comment_id, text FROM comments 
        WHERE sentiment_processed = FALSE
        LIMIT %s
        FOR UPDATE SKIP LOCKED
        """, (BATCH_SIZE,))

        batch = cur.fetchall()
        if not batch:
            return False

        # Process each comment
        for comment_id, text in batch:
            try:
                analysis = analyze_sentiment(text)

                # Insert sentiment analysis
                cur.execute("""
                INSERT INTO sentiment_analysis (comment_id, sentiment, confidence)
                VALUES (%s, %s, %s)
                ON CONFLICT (comment_id) DO UPDATE
                SET sentiment = EXCLUDED.sentiment,
                    confidence = EXCLUDED.confidence,
                    processed_at = NOW()
                """, (comment_id, analysis['sentiment'], analysis['confidence']))

                # Mark as processed
                cur.execute("""
                UPDATE comments 
                SET sentiment_processed = TRUE 
                WHERE comment_id = %s
                """, (comment_id,))

                conn.commit()
            except Exception as e:
                print(f"Error processing sentiment for comment {comment_id}: {str(e)}")
                conn.rollback()

        return True

def ensure_tables(conn):
    """Ensure required tables exist"""
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            id SERIAL PRIMARY KEY,
            comment_id TEXT REFERENCES comments(comment_id),
            sentiment TEXT,
            confidence FLOAT,
            processed_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(comment_id)
        )
        """)

        # Add column if not exists
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_name='comments' AND column_name='sentiment_processed') THEN
                ALTER TABLE comments ADD COLUMN sentiment_processed BOOLEAN DEFAULT FALSE;
            END IF;
        END $$;
        """)
        conn.commit()


def main():
    print("Starting sentiment processor...")
    conn = connect_db()
    ensure_tables(conn)

    try:
        processed = 0
        while True:
            if process_batch(conn):
                processed += BATCH_SIZE
                print(f"Processed {processed} comments...")
                time.sleep(1)  # Small delay between batches
            else:
                print("No more comments to process. Waiting...")
                time.sleep(60)  # Wait a minute before checking again
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        conn.close()


if __name__ == "__main__":
    main()