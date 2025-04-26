import os
import json
import psycopg2
from psycopg2 import sql
from datetime import datetime
import glob


def connect_db():
    return psycopg2.connect(
        dbname="comments_db",
        user="comment_user",
        password="comment_password",
        host="db"
    )


def setup_database(conn):
    with conn.cursor() as cur:
        # Create tables
        cur.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id SERIAL PRIMARY KEY,
            video_id INTEGER UNIQUE NOT NULL,
            title TEXT,
            description TEXT,
            url TEXT,
            download_link TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS video_tags (
            video_id INTEGER REFERENCES videos(video_id),
            tag_id INTEGER REFERENCES tags(id),
            PRIMARY KEY (video_id, tag_id)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            profile_url TEXT,
            avatar_url TEXT,
            avatar_local_path TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id SERIAL PRIMARY KEY,
            comment_id TEXT UNIQUE NOT NULL,
            video_id INTEGER REFERENCES videos(video_id),
            user_id INTEGER REFERENCES users(id),
            parent_id TEXT REFERENCES comments(comment_id),
            text TEXT NOT NULL,
            timestamp TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            upvotes INTEGER DEFAULT 0,
            has_more_replies BOOLEAN DEFAULT FALSE,
            reply_count INTEGER DEFAULT 0
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS mentions (
            id SERIAL PRIMARY KEY,
            comment_id TEXT REFERENCES comments(comment_id),
            mentioned_username TEXT,
            FOREIGN KEY (mentioned_username) REFERENCES users(username)
        )
        """)

        conn.commit()


def process_json_files(conn, data_folder):
    for filepath in glob.glob(f"{data_folder}/*.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for video_id, video_data in data.items():
                try:
                    process_video(conn, video_id, video_data)
                except Exception as e:
                    print(f"Error processing video {video_id} in file {filepath}: {str(e)}")
                    conn.rollback()


def process_video(conn, video_id, video_data):
    with conn.cursor() as cur:
        # Insert video metadata
        metadata = video_data['metadata']
        cur.execute("""
        INSERT INTO videos (video_id, title, description, url, download_link)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO NOTHING
        RETURNING video_id
        """, (int(video_id), metadata.get('title'), metadata.get('description'),
              metadata.get('url'), metadata.get('download_link')))

        # Insert tags
        for tag_name in metadata.get('tags', []):
            cur.execute("""
            INSERT INTO tags (name)
            VALUES (%s)
            ON CONFLICT (name) DO NOTHING
            RETURNING id
            """, (tag_name,))

            tag_id = cur.fetchone()
            if tag_id:
                tag_id = tag_id[0]
                cur.execute("""
                INSERT INTO video_tags (video_id, tag_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """, (int(video_id), tag_id))

        # Process comments
        for comment in video_data.get('comments', []):
            try:
                process_comment(conn, comment, int(video_id))
            except Exception as e:
                print(f"Error processing comment {comment.get('id')} in video {video_id}: {str(e)}")
                conn.rollback()

        conn.commit()


def process_comment(conn, comment, video_id, parent_id=None):
    with conn.cursor() as cur:
        # Insert user
        user = comment.get('user', {})
        cur.execute("""
        INSERT INTO users (username, profile_url, avatar_url, avatar_local_path)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (username) DO NOTHING
        RETURNING id
        """, (user.get('username'), user.get('profile_url'),
              user.get('avatar'), user.get('avatar_local')))

        user_result = cur.fetchone()
        if user_result:
            user_id = user_result[0]
        else:
            cur.execute("SELECT id FROM users WHERE username = %s", (user.get('username'),))
            user_id = cur.fetchone()[0]

        # Insert comment with default values for missing fields
        cur.execute("""
        INSERT INTO comments (
            comment_id, video_id, user_id, parent_id, text, 
            timestamp, upvotes, has_more_replies, reply_count
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (comment_id) DO NOTHING
        """, (
            comment.get('id'),
            video_id,
            user_id,
            parent_id,
            comment.get('text'),
            comment.get('timestamp'),
            comment.get('upvotes', 0),
            comment.get('has_more_replies', False),
            comment.get('reply_count', 0)
        ))

        # Process replies if they exist
        for reply in comment.get('replies', []):
            try:
                process_comment(conn, reply, video_id, comment.get('id'))

                # Process mentions if they exist
                for mention in reply.get('mentions', []):
                    # First ensure the mentioned user exists
                    mentioned_username = mention.strip()
                    cur.execute("""
                    INSERT INTO users (username)
                    VALUES (%s)
                    ON CONFLICT (username) DO NOTHING
                    """, (mentioned_username,))

                    # Then insert the mention
                    cur.execute("""
                    INSERT INTO mentions (comment_id, mentioned_username)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """, (reply.get('id'), mentioned_username))
            except Exception as e:
                print(f"Error processing reply {reply.get('id')}: {str(e)}")
                conn.rollback()

        conn.commit()


if __name__ == "__main__":
    print("Starting data import...")
    conn = connect_db()
    setup_database(conn)

    data_folder = "/app/data"  # Mounted volume from host
    if os.path.exists(data_folder):
        print(f"Processing JSON files in {data_folder}...")
        process_json_files(conn, data_folder)
    else:
        print(f"Data folder not found: {data_folder}")

    conn.close()
    print("Data import completed.")

