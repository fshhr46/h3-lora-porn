#!/usr/bin/env python3
"""
素材收集脚本 - 从 YouTube/Twitter/Reddit 收集视频素材
支持自动下载、抽帧、裁剪

使用方法:
    python collect.py --source youtube --keywords "beautiful woman,bikini" --count 20
    python collect.py --source twitter --keywords "anime girl" --count 20
    python collect.py --source reddit --subreddit "facelesspods" --count 20
"""

import argparse
import os
import sys
import time
import hashlib
import subprocess
import re
from pathlib import Path
from urllib.parse import urlparse

try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False
    print("⚠️  yt_dlp 未安装，YouTube 下载功能不可用")
    print("   安装: pip install yt-dlp")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  OpenCV 未安装，视频处理功能不可用")
    print("   安装: pip install opencv-python")


# ============================================================
# YouTube 收集器
# ============================================================

class YouTubeCollector:
    """从 YouTube 收集视频素材"""

    def __init__(self, output_dir, quality='bestvideo'):
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.ydl_opts = {
            'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'noplaylist': True,
            'max_downloads': 1,
            'writethumbnail': True,
            'skip_download': False,
        }

    def search_and_download(self, keywords, count=20):
        """搜索并下载视频"""
        if not HAS_YTDLP:
            print("❌ yt_dlp 未安装，无法下载 YouTube 视频")
            return 0

        downloaded = 0
        for keyword in keywords.split(','):
            keyword = keyword.strip()
            if not keyword:
                continue

            print(f"\n🔍 搜索 YouTube: {keyword}")
            search_query = f"ytsearch{count}:{keyword} slow motion beautiful"

            self.ydl_opts['outtmpl'] = str(self.output_dir / keyword.replace(' ', '_') / '%(id)s.%(ext)s')

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(search_query, download=False)
                    videos = info.get('entries', [])

                    for video in videos[:count]:
                        if downloaded >= count:
                            break

                        video_id = video.get('id')
                        if not video_id:
                            continue

                        # 检查是否已下载
                        video_path = self.output_dir / keyword.replace(' ', '_') / f"{video_id}.mp4"
                        if video_path.exists():
                            print(f"  ⏭️  跳过已下载: {video_id}")
                            downloaded += 1
                            continue

                        # 下载视频
                        self.ydl_opts['outtmpl'] = str(self.output_dir / keyword.replace(' ', '_') / '%(id)s.%(ext)s')
                        try:
                            ydl.download([video.get('webpage_url', '')])
                            print(f"  ✅ 已下载: {video_id}")
                            downloaded += 1
                        except Exception as e:
                            print(f"  ❌ 下载失败 {video_id}: {e}")

                        time.sleep(2)  # 避免 rate limit

                except Exception as e:
                    print(f"  ❌ 搜索失败 '{keyword}': {e}")

            time.sleep(1)

        return downloaded


# ============================================================
# Twitter/X 收集器
# ============================================================

class TwitterCollector:
    """从 Twitter 收集视频素材"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def search_and_download(self, keywords, count=20):
        """搜索并下载 Twitter 视频"""
        if not HAS_YTDLP:
            print("❌ yt_dlp 未安装，无法下载 Twitter 视频")
            return 0

        downloaded = 0
        for keyword in keywords.split(','):
            keyword = keyword.strip()
            if not keyword:
                continue

            print(f"\n🔍 搜索 Twitter: {keyword}")
            search_query = f"ytsearch{count}:site:twitter.com {keyword} slow motion"

            ydl_opts = {
                'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
                'outtmpl': str(self.output_dir / 'twitter' / '%(id)s.%(ext)s'),
                'noplaylist': True,
                'max_downloads': 1,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(search_query, download=False)
                    videos = info.get('entries', [])

                    for video in videos[:count]:
                        if downloaded >= count:
                            break

                        video_id = video.get('id')
                        url = video.get('webpage_url', '')

                        if 'twitter.com' not in url and 'x.com' not in url:
                            continue

                        try:
                            ydl.download([url])
                            print(f"  ✅ 已下载 Twitter 视频")
                            downloaded += 1
                        except Exception as e:
                            print(f"  ❌ 下载失败: {e}")

                        time.sleep(1)

                except Exception as e:
                    print(f"  ❌ 搜索失败: {e}")

        return downloaded


# ============================================================
# Reddit 收集器
# ============================================================

class RedditCollector:
    """从 Reddit 收集视频素材"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def search_and_download(self, subreddit, count=20):
        """从指定 subreddit 收集视频"""
        if not HAS_YTDLP:
            print("❌ yt_dlp 未安装，无法下载 Reddit 视频")
            return 0

        downloaded = 0
        search_query = f"ytsearch{count}:reddit.com/r/{subreddit}"

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
            'outtmpl': str(self.output_dir / 'reddit' / '%(id)s.%(ext)s'),
            'noplaylist': True,
            'max_downloads': 1,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(search_query, download=False)
                videos = info.get('entries', [])

                for video in videos[:count]:
                    if downloaded >= count:
                        break

                    video_id = video.get('id')
                    url = video.get('webpage_url', '')

                    if 'reddit.com' not in url and 'redgifs.com' not in url:
                        continue

                    try:
                        ydl.download([url])
                        print(f"  ✅ 已下载 Reddit 视频")
                        downloaded += 1
                    except Exception as e:
                        print(f"  ❌ 下载失败: {e}")

                    time.sleep(1)

            except Exception as e:
                print(f"  ❌ 搜索失败: {e}")

        return downloaded


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='视频素材收集器')
    parser.add_argument('--source', type=str, required=True,
                        choices=['youtube', 'twitter', 'reddit'],
                        help='数据源')
    parser.add_argument('--keywords', type=str, default='beautiful woman,slow motion',
                        help='搜索关键词（逗号分隔）')
    parser.add_argument('--subreddit', type=str, default='facelesspods',
                        help='Reddit subreddit 名称')
    parser.add_argument('--count', type=int, default=20,
                        help='每个关键词下载数量')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='输出目录')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 素材收集器 - 源: {args.source}")
    print(f"   输出: {output_dir}")
    print("=" * 60)

    if args.source == 'youtube':
        collector = YouTubeCollector(output_dir)
        collected = collector.search_and_download(args.keywords, args.count)
    elif args.source == 'twitter':
        collector = TwitterCollector(output_dir)
        collected = collector.search_and_download(args.keywords, args.count)
    elif args.source == 'reddit':
        collector = RedditCollector(output_dir)
        collected = collector.search_and_download(args.subreddit, args.count)
    else:
        print(f"❌ 不支持的源: {args.source}")
        return

    print("\n" + "=" * 60)
    print(f"✅ 共下载 {collected} 个视频到 {output_dir}")
    print("\n下一步: 运行预处理脚本")
    print("   python preprocess.py --input data/raw --output data/cropped")


if __name__ == '__main__':
    main()
