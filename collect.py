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
# RedGifs 收集器（成人短视频最佳来源）
# ============================================================

class RedGifsCollector:
    """从 RedGifs 收集成人短视频素材"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / 'redgifs'
        self.api_base = 'https://api.redgifs.com/v3'
        self.token = None

    def _authenticate(self):
        """获取 RedGifs API token"""
        try:
            import requests
            resp = requests.post(f'{self.api_base}/tokens', headers={'Content-Type': 'application/json'})
            if resp.status_code == 200:
                self.token = resp.json().get('token')
                print("  ✅ RedGifs API 认证成功")
        except Exception as e:
            print(f"  ⚠️  RedGifs API 认证失败: {e}")

    def search_and_download(self, keywords, count=30):
        """搜索并下载 RedGifs 视频"""
        if not HAS_YTDLP:
            print("❌ yt_dlp 未安装")
            return 0

        self._authenticate()
        if not self.token:
            print("  ❌ 无法获取 API token")
            return 0

        downloaded = 0
        headers = {'Authorization': f'Bearer {self.token}'}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for keyword in keywords.split(','):
            keyword = keyword.strip()
            if not keyword:
                continue

            print(f"\n🔍 搜索 RedGifs: {keyword}")
            search_query = f"ytsearch{count}:redgifs.com {keyword}"

            ydl_opts = {
                'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
                'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
                'noplaylist': True,
                'max_downloads': 1,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(search_query, download=False)
                    gifs = info.get('entries', [])

                    for gif in gifs[:count]:
                        if downloaded >= count:
                            break

                        gif_id = gif.get('id')
                        url = gif.get('webpage_url', '')

                        if 'redgifs.com' not in url:
                            continue

                        try:
                            ydl.download([url])
                            print(f"  ✅ 已下载 RedGifs")
                            downloaded += 1
                        except Exception as e:
                            print(f"  ❌ 下载失败: {e}")

                        time.sleep(1)

                except Exception as e:
                    print(f"  ❌ 搜索失败: {e}")

        return downloaded


# ============================================================
# Xvideos 收集器
# ============================================================

class XvideosCollector:
    """从 Xvideos 收集视频素材"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / 'xvideos'
        self.ydl_opts = {
            'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'noplaylist': True,
            'max_downloads': 1,
        }

    def search_and_download(self, keywords, count=20):
        """搜索并下载 Xvideos 视频"""
        if not HAS_YTDLP:
            print("❌ yt_dlp 未安装")
            return 0

        downloaded = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for keyword in keywords.split(','):
            keyword = keyword.strip()
            if not keyword:
                continue

            print(f"\n🔍 搜索 Xvideos: {keyword}")
            search_query = f"ytsearch{count}:xvideos.com {keyword}"

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(search_query, download=False)
                    videos = info.get('entries', [])

                    for video in videos[:count]:
                        if downloaded >= count:
                            break

                        url = video.get('webpage_url', '')
                        if 'xvideos.com' not in url:
                            continue

                        try:
                            ydl.download([url])
                            print(f"  ✅ 已下载 Xvideos")
                            downloaded += 1
                        except Exception as e:
                            print(f"  ❌ 下载失败: {e}")

                        time.sleep(2)

                except Exception as e:
                    print(f"  ❌ 搜索失败: {e}")

        return downloaded


# ============================================================
# SpankWire 收集器（短片段素材）
# ============================================================

class SpankWireCollector:
    """从 SpankWire 收集短片段素材"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / 'spankwire'
        self.ydl_opts = {
            'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'noplaylist': True,
            'max_downloads': 1,
        }

    def search_and_download(self, keywords, count=20):
        """搜索并下载 SpankWire 视频"""
        if not HAS_YTDLP:
            print("❌ yt_dlp 未安装")
            return 0

        downloaded = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for keyword in keywords.split(','):
            keyword = keyword.strip()
            if not keyword:
                continue

            print(f"\n🔍 搜索 SpankWire: {keyword}")
            search_query = f"ytsearch{count}:spankwire.com {keyword}"

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(search_query, download=False)
                    videos = info.get('entries', [])

                    for video in videos[:count]:
                        if downloaded >= count:
                            break

                        url = video.get('webpage_url', '')
                        if 'spankwire.com' not in url:
                            continue

                        try:
                            ydl.download([url])
                            print(f"  ✅ 已下载 SpankWire")
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
                        choices=['youtube', 'twitter', 'reddit', 'redgifs', 'xvideos', 'spankwire'],
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
    elif args.source == 'redgifs':
        collector = RedGifsCollector(output_dir)
        collected = collector.search_and_download(args.keywords, args.count)
    elif args.source == 'xvideos':
        collector = XvideosCollector(output_dir)
        collected = collector.search_and_download(args.keywords, args.count)
    elif args.source == 'spankwire':
        collector = SpankWireCollector(output_dir)
        collected = collector.search_and_download(args.keywords, args.count)
    else:
        print(f"❌ 不支持的源: {args.source}")
        return

    print("\n" + "=" * 60)
    print(f"✅ 共下载 {collected} 个视频到 {output_dir}")
    print("\n下一步: 运行预处理脚本")
    print("   python preprocess.py --input data/raw --output data/cropped")


if __name__ == '__main__':
    main()
