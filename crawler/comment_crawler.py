import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommentFullCrawler:
    def __init__(self, base_url, cookies, headers, download_videos=False):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(headers)
        self.cookies = cookies
        self.download_videos = download_videos
        self.failed_ids = set()
        self.successful_ids = set()
        self.profile_pics_dir = "profile_pics"
        os.makedirs(self.profile_pics_dir, exist_ok=True)
        if download_videos:
            os.makedirs("videos", exist_ok=True)

    def get_video_page(self, content_id):
        """Get the video page HTML"""
        url = f"{self.base_url}/video/{content_id}"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
            return None
        except Exception as e:
            logger.error(f"Error getting video page {content_id}: {str(e)}")
            return None

    def get_video_metadata(self, content_id):
        """Extract video metadata from the video page with multiple quality download link attempts"""
        html = self.get_video_page(content_id)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        try:
            title = soup.find('meta', {'property': 'og:title'}).get('content', '')
            description = soup.find('meta', {'property': 'og:description'}).get('content', '')
            url = soup.find('meta', {'property': 'og:url'}).get('content', '')

            tags = []
            for tag in soup.find_all('meta', {'property': 'video:tag'}):
                if tag.get('content'):
                    tags.append(tag.get('content'))

            download_link = None
            if self.download_videos:
                # Try multiple quality options in order of preference
                for quality in ['1080p', 'HD', 'SD']:
                    link_tag = soup.find('a', href=re.compile(f'download\.php\?id=\d+&label={quality}'))
                    if link_tag and link_tag.get('href'):
                        download_link = urljoin(self.base_url, link_tag.get('href'))
                        logger.info(f"Found {quality} download link for video {content_id}")
                        break

                # If no link found with standard pattern, try alternative approach
                if not download_link:
                    link_tag = soup.find('a', href=re.compile(r'download\.php\?id=\d+'))
                    if link_tag and link_tag.get('href'):
                        download_link = urljoin(self.base_url, link_tag.get('href'))
                        logger.info(f"Found generic download link for video {content_id}")

            return {
                "id": content_id,
                "title": title,
                "description": description,
                "url": url,
                "tags": tags,
                "download_link": download_link
            }
        except Exception as e:
            logger.error(f"Error parsing video metadata for {content_id}: {str(e)}")
            return None

    def download_video(self, content_id, video_metadata):
        """Download the video file with multiple quality fallbacks"""
        if not video_metadata.get('download_link'):
            logger.error(f"No download link found for video {content_id}")
            return None

        formatted_title = re.sub(r'[^\w\s-]', '', video_metadata['title']).replace(' ', '_')
        mp4_file_name = f"videos/{content_id}_{formatted_title}.mp4"

        logger.info(f"Starting download for video {content_id} to {mp4_file_name}")

        try:
            headers = {}
            if os.path.exists(mp4_file_name):
                file_size = os.path.getsize(mp4_file_name)
                headers = {'Range': f'bytes={file_size}-'}
            else:
                file_size = 0

            # Try the primary download link first
            response = self.session.get(
                video_metadata['download_link'],
                headers=headers,
                stream=True,
                timeout=30
            )

            # If the primary link fails (e.g., 403), try alternative qualities
            if response.status_code == 403:
                logger.warning(f"Primary download link failed, trying alternative qualities...")
                base_url = video_metadata['download_link'].split('&label=')[0]
                for quality in ['1080p', 'HD', 'SD']:
                    alt_url = f"{base_url}&label={quality}"
                    logger.info(f"Trying {quality} quality...")
                    response = self.session.get(alt_url, headers=headers, stream=True, timeout=30)
                    if response.status_code == 200:
                        video_metadata['download_link'] = alt_url
                        logger.info(f"Successfully switched to {quality} quality")
                        break

            response.raise_for_status()
            total_length = response.headers.get('content-length')

            if total_length is None:
                logger.error(f"Cannot determine file size for video {content_id}")
                return None

            total_length = int(total_length) + file_size
            total_length_mb = total_length / (1024 * 1024)
            downloaded = file_size
            prev_crc32 = 0
            bar_length = 50
            update_size = 1024 * 1024

            mode = 'ab' if file_size else 'wb'
            with open(mp4_file_name, mode) as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        prev_crc32 = zlib.crc32(chunk, prev_crc32)

                        if downloaded % update_size < 8192 or downloaded >= total_length:
                            downloaded_mb = downloaded / (1024 * 1024)
                            filled_length = int(bar_length * downloaded / total_length)
                            bar = '█' * filled_length + '-' * (bar_length - filled_length)
                            progress = (downloaded / total_length) * 100
                            logger.info(
                                f"\r[{bar}] {progress:.2f}% ({downloaded_mb:.2f} MB / {total_length_mb:.2f} MB)")

                logger.info(f"\nDownload completed for video {content_id}")

            crc32_hash = "%08X" % (prev_crc32 & 0xFFFFFFFF)
            return crc32_hash

        except Exception as e:
            logger.error(f"Error downloading video {content_id}: {str(e)}")
            return None
    def download_profile_pic(self, url):
        """Download profile picture if it doesn't exist"""
        if not url or 'nopic-' in url:  # Skip default profile pics
            return None

        filename = os.path.basename(url)
        local_path = os.path.join(self.profile_pics_dir, filename)

        if os.path.exists(local_path):
            return local_path

        try:
            response = self.session.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return local_path
        except Exception as e:
            logger.error(f"Error downloading profile pic {url}: {str(e)}")

        return None

    def load_comments(self, content_id, content_type="video", page=1, order="newest"):
        """Load comments for a specific content"""
        url = f"{self.base_url}/ajax/load_comments"
        data = {
            "id": content_id,
            "type": content_type,
            "page": page,
            "order": order
        }

        try:
            response = self.session.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
        except Exception as e:
            logger.error(f"Error loading comments for {content_id}: {str(e)}")
            return None
        return None

    def load_replies(self, comment_id, content_type="video", page=1, order="newest"):
        """Load replies for a specific comment"""
        url = f"{self.base_url}/ajax/load_replies"
        data = {
            "id": comment_id,
            "type": content_type,
            "page": page,
            "order": order
        }

        try:
            response = self.session.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error loading replies for comment {comment_id}: {str(e)}")
            return None
        return None

    def parse_comment(self, comment_html):
        """Parse a single comment HTML element into a dictionary"""
        soup = BeautifulSoup(comment_html, 'html.parser')

        try:
            comment = {
                "id": soup.find('div', class_='comment-item').get('id', '').replace('comment_', ''),
                "user": {
                    "username": soup.find('a', class_='comment-username').text if soup.find('a',
                                                                                            class_='comment-username') else None,
                    "profile_url": soup.find('a', class_='comment-username').get('href') if soup.find('a',
                                                                                                      class_='comment-username') else None,
                    "avatar": None,  # Will be filled later
                    "avatar_local": None  # Will be filled later
                },
                "timestamp": soup.find('span', class_='comment-add-time').text if soup.find('span',
                                                                                            class_='comment-add-time') else None,
                "text": soup.find('div', class_='comment-text').get_text(separator='\n').strip() if soup.find('div',
                                                                                                              class_='comment-text') else None,
                "upvotes": int(
                    soup.find('span', id=lambda x: x and x.startswith('comment_rate_video_')).text) if soup.find('span',
                                                                                                                 id=lambda
                                                                                                                     x: x and x.startswith(
                                                                                                                     'comment_rate_video_')) else 0,
                "replies": [],
                "has_more_replies": False,
                "reply_count": 0
            }

            # Get and download profile picture
            user_div = soup.find('div', class_='comment-user')
            if user_div and user_div.find('img'):
                avatar_url = user_div.find('img').get('src')
                comment["user"]["avatar"] = avatar_url
                if avatar_url:
                    local_path = self.download_profile_pic(avatar_url)
                    comment["user"]["avatar_local"] = local_path

            # Check if there are replies available
            replies_container = soup.find('div', class_='comment-replies')
            if replies_container:
                more_replies = replies_container.find('a', id=lambda x: x and x.startswith('replies_show_more_video_'))
                if more_replies:
                    comment["has_more_replies"] = True
                    try:
                        comment["reply_count"] = int(
                            more_replies.find('span', id=lambda x: x and x.startswith('replies_total_video_')).text)
                    except:
                        pass

            return comment
        except Exception as e:
            logger.error(f"Error parsing comment HTML: {str(e)}")
            return None

    def parse_reply(self, reply_html, parent_id):
        """Parse a single reply HTML element into a dictionary"""
        soup = BeautifulSoup(reply_html, 'html.parser')

        try:
            comment_div = soup.find('div', class_='comment-item')
            username_tag = soup.find('a', class_='comment-username')
            user_div = soup.find('div', class_='comment-user')
            avatar_img = user_div.find('img') if user_div else None
            timestamp_tag = soup.find('span', class_='comment-add-time')
            text_div = soup.find('div', class_='comment-text')
            upvotes_tag = soup.find('span', id=lambda x: x and x.startswith('comment_rate_video_'))

            reply = {
                "id": comment_div.get('id', '').replace('comment_', '') if comment_div else None,
                "parent_id": parent_id,
                "user": {
                    "username": username_tag.text if username_tag else None,
                    "profile_url": username_tag.get('href') if username_tag else None,
                    "avatar": avatar_img.get('src') if avatar_img else None,
                    "avatar_local": None  # Will be filled later
                },
                "timestamp": timestamp_tag.text if timestamp_tag else None,
                "text": text_div.get_text(separator='\n').strip() if text_div else None,
                "upvotes": int(upvotes_tag.text) if upvotes_tag and upvotes_tag.text.isdigit() else 0,
                "mentions": self.extract_mentions(text_div.get_text()) if text_div else []
            }

            # Download profile picture for reply
            if reply["user"]["avatar"]:
                local_path = self.download_profile_pic(reply["user"]["avatar"])
                reply["user"]["avatar_local"] = local_path

            return reply

        except Exception as e:
            logger.error(f"Error parsing reply HTML: {str(e)}")
            return None

    def extract_mentions(self, text):
        """Extract @mentions from comment text"""
        mentions = []
        if text and '@' in text:
            words = text.split()
            for word in words:
                if word.startswith('@'):
                    mentions.append(word[1:])  # Remove @ symbol
        return mentions

    def get_content_data(self, content_id, content_type="video", max_comment_pages=10, max_reply_pages=10, delay=5):
        """Get all data for a single content (metadata + comments)"""
        # Get video metadata
        video_metadata = self.get_video_metadata(content_id)
        if not video_metadata:
            return None

        # Get comments
        all_comments = []
        page = 1

        while True:
            logger.info(f"Loading comments page {page} for content {content_id}...")
            response = self.load_comments(content_id, content_type, page)

            if not response:
                break

            if not response.get('code'):
                if page == 1:  # No comments at all for this content
                    break
                else:  # No more comments
                    break

            soup = BeautifulSoup(response['code'], 'html.parser')
            comment_items = soup.find_all('div', class_='comment-item')

            if not comment_items:
                break

            for comment_html in comment_items:
                comment = self.parse_comment(str(comment_html))
                if not comment:
                    continue

                # Load replies if available
                if comment["has_more_replies"]:
                    reply_page = 1
                    while True:
                        logger.info(f"Loading replies page {reply_page} for comment {comment['id']}...")
                        reply_response = self.load_replies(comment['id'], content_type, reply_page)

                        if not reply_response or not reply_response.get('code'):
                            break

                        reply_soup = BeautifulSoup(reply_response['code'], 'html.parser')
                        reply_items = reply_soup.find_all('div', class_='comment-item')

                        if not reply_items:
                            break

                        for reply_html in reply_items:
                            reply = self.parse_reply(str(reply_html), comment['id'])
                            if reply:
                                comment['replies'].append(reply)

                        # Check if there are more replies to load
                        if reply_response.get('more_replies', 0) <= 0 or reply_page >= max_reply_pages:
                            break

                        reply_page += 1
                        time.sleep(delay)  # Be polite

                all_comments.append(comment)
                time.sleep(delay / 2)  # Short delay between comments

            # Check if there are more comments to load
            if response.get('more_comments', 0) <= 0 or page >= max_comment_pages:
                break

            page += 1
            time.sleep(delay)  # Be polite

        # Combine all data
        # Transform the structure
        video_id = str(content_id)
        result = {
            video_id: {
                "metadata": video_metadata,
                "comments": all_comments
            }
        }
        """
        result = {
            "metadata": video_metadata,
            "comments": all_comments
        }
        """

        # Optionally download video
        if self.download_videos and video_metadata.get('download_link'):
            crc32_hash = self.download_video(content_id, video_metadata)
            if crc32_hash:
                result["metadata"]["download_crc32"] = crc32_hash

        return result

    def crawl_content_range(self, start_id, end_id, content_type="video", max_workers=5):
        """Crawl a range of content IDs with multithreading"""
        results = {}
        total_processed = 0
        successful = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.get_content_data, content_id, content_type): content_id
                for content_id in range(start_id, end_id + 1)
            }

            for future in as_completed(future_to_id):
                content_id = future_to_id[future]
                try:
                    content_data = future.result()
                    if content_data:  # Only save if we got data
                        results[content_id] = content_data
                        successful += 1
                        self.successful_ids.add(content_id)

                        # Save individual JSON file for this video
                        filename = f"data/{content_id}.json"
                        os.makedirs("data", exist_ok=True)
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(content_data, f, ensure_ascii=False, indent=2)

                        logger.info(f"Successfully processed content {content_id}")
                    else:
                        self.failed_ids.add(content_id)
                except Exception as e:
                    self.failed_ids.add(content_id)
                    logger.error(f"Error processing content {content_id}: {str(e)}")

                total_processed += 1
                if total_processed % 100 == 0:
                    logger.info(f"Progress: {total_processed}/{end_id - start_id + 1} - Successful: {successful}")

        logger.info(f"Crawling completed. Successful: {successful}, Failed: {len(self.failed_ids)}")
        return results


if __name__ == "__main__":
    base_url = ""

    # Configure with your cookies and headers
    cookies = {
        "MEGNETTEMAODATOT": "BAZDMARMEG",
        "AVS": "",
        "cf_clearance": "",
        "remember": ""
    }

    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "en-US,en;q=0.6",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "origin": base_url,
        "priority": "u=1, i",
        "referer": base_url,
        "sec-ch-ua": '"Brave";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        "sec-ch-ua-arch": '"x86"',
        "sec-ch-ua-bitness": '"64"',
        "sec-ch-ua-full-version-list": '"Brave";v="135.0.0.0", "Not-A.Brand";v="8.0.0.0", "Chromium";v="135.0.0.0"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": '""',
        "sec-ch-ua-platform": '"Windows"',
        "sec-ch-ua-platform-version": '"15.0.0"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "x-requested-with": "XMLHttpRequest"
    }

    # Initialize crawler (set download_videos=True to enable video downloading)
    crawler = CommentFullCrawler(base_url, cookies, headers, download_videos=False)

    # Crawl a range of content IDs (adjust range as needed)
    # delay = 5
    start_id = 0
    end_id = 20000  # Adjust this based on how many IDs you want to check

    # Start crawling (using 4 threads)
    all_data = crawler.crawl_content_range(
        start_id=start_id,
        end_id=end_id,
        content_type="video",
        max_workers=2
    )

    # Save final combined results (optional)
    with open(f"all_data_{start_id}_to_{end_id}.json", 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

