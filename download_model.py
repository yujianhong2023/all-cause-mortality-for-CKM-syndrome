#!/usr/bin/env python3
"""
123云盘模型文件下载脚本（增强版）
支持无提取码分享链接，内置多重解析方案
"""

import os
import re
import sys
import json
import time
import requests
from urllib.parse import unquote


def generate_random_number(length):
    """生成指定长度的随机数字字符串"""
    import random
    return ''.join(str(random.randint(0, 9)) for _ in range(length))


def parse_123pan_v1(share_url):
    """
    方案一：使用 /b/api/share/get 接口解析（参考自代码狗[reference:0]）
    适用于无提取码的分享链接
    """
    # 提取 shareKey
    match = re.search(r'/123pan/([^?]+)', share_url)
    if not match:
        raise Exception("无法解析分享链接")
    share_key = match.group(1)
    print(f"📌 分享Key: {share_key}")

    # 生成随机数和时间戳
    timestamp = int(time.time())
    rand1 = generate_random_number(10)
    rand2 = generate_random_number(7)
    rand3 = generate_random_number(10)

    # 构建请求参数
    params = {
        rand1: f"{timestamp}-{rand2}-{rand3}",
        "limit": 100,
        "next": 0,
        "orderBy": "file_name",
        "orderDirection": "asc",
        "shareKey": share_key,
        "SharePwd": "",  # 无提取码
        "ParentFileId": 0,
        "Page": 1,
        "event": "homeListFile",
        "operateType": 1
    }

    info_url = "https://www.123pan.com/b/api/share/get"
    print("🔐 正在获取文件信息...")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.123pan.com/',
    })

    resp = session.get(info_url, params=params, timeout=30)

    if resp.status_code != 200:
        raise Exception(f"请求失败，状态码: {resp.status_code}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise Exception("解析响应失败")

    if data.get('code') != 0:
        raise Exception(f"获取文件信息失败: {data.get('message', '未知错误')}")

    file_list = data.get('data', {}).get('InfoList', [])
    if not file_list:
        raise Exception("未找到文件")

    file_info = file_list[0]
    file_name = file_info.get('FileName', 'rsf_model.pkl')
    file_id = file_info.get('FileId')
    s3key_flag = file_info.get('S3KeyFlag')
    size = file_info.get('Size')
    etag = file_info.get('Etag')

    print(f"📁 目标文件: {file_name}")

    # 获取下载直链
    download_params = {
        rand1: f"{timestamp}-{rand2}-{rand3}"
    }
    download_url = "https://www.123pan.com/b/api/share/download/info"

    payload = {
        "ShareKey": share_key,
        "FileId": file_id,
        "S3KeyFlag": s3key_flag,
        "Size": size,
        "Etag": etag
    }

    print("⏳ 正在获取下载直链...")
    resp = session.post(download_url, params=download_params, json=payload, timeout=30)

    if resp.status_code != 200:
        raise Exception(f"获取下载链接失败，状态码: {resp.status_code}")

    # 解析返回的 base64 编码的直链
    text = resp.text
    match = re.search(r'params=(.+?)(?:\\u|&|$)', text)
    if not match:
        # 尝试直接解析 JSON
        try:
            json_data = resp.json()
            direct_url = json_data.get('data', {}).get('downloadUrl')
            if direct_url:
                return direct_url, file_name
        except:
            pass
        raise Exception("无法解析下载链接")

    import base64
    encoded = match.group(1)
    direct_url = base64.b64decode(encoded).decode('utf-8')
    return direct_url, file_name


def parse_123pan_v2(share_url):
    """
    方案二：使用第三方解析 API（备用方案）[reference:1][reference:2]
    """
    api_url = "http://api.nonebot.top/api/v1/cloud/pan123/parse"

    payload = {
        "shareUrl": share_url,
        "password": ""  # 无提取码
    }

    print("🔄 尝试第三方解析...")
    resp = requests.post(api_url, json=payload, timeout=30)

    if resp.status_code != 200:
        raise Exception(f"第三方API请求失败: {resp.status_code}")

    data = resp.json()
    if data.get('code') != 0:
        raise Exception(f"第三方解析失败: {data.get('msg', '未知错误')}")

    file_name = data.get('data', {}).get('fileName', 'rsf_model.pkl')
    direct_url = data.get('data', {}).get('downloadUrl')

    if not direct_url:
        raise Exception("未能获取到下载链接")

    return direct_url, file_name


def parse_123pan_v3(share_url):
    """
    方案三：使用 CloudDiskAnalysis 解析 API
    """
    api_url = "https://cloud.humorously.cn/api/123pan.php"

    params = {
        "link": share_url,
        "pwd": ""  # 无提取码
    }

    print("🔄 尝试备用解析服务...")
    resp = requests.get(api_url, params=params, timeout=30)

    if resp.status_code != 200:
        raise Exception(f"备用API请求失败: {resp.status_code}")

    data = resp.json()
    if data.get('code') != 200:
        raise Exception(f"备用解析失败: {data.get('msg', '未知错误')}")

    file_name = data.get('data', {}).get('name', 'rsf_model.pkl')
    direct_url = data.get('data', {}).get('url')

    if not direct_url:
        raise Exception("未能获取到下载链接")

    return direct_url, file_name


def get_direct_url(share_url):
    """
    依次尝试多种解析方案
    """
    errors = []

    # 方案一：官方接口解析
    try:
        print("📌 尝试方案一：官方接口解析")
        return parse_123pan_v1(share_url)
    except Exception as e:
        errors.append(f"方案一失败: {e}")
        print(f"   ⚠️ {e}")

    # 方案二：第三方 API
    try:
        print("📌 尝试方案二：第三方 API")
        return parse_123pan_v2(share_url)
    except Exception as e:
        errors.append(f"方案二失败: {e}")
        print(f"   ⚠️ {e}")

    # 方案三：备用解析服务
    try:
        print("📌 尝试方案三：备用解析服务")
        return parse_123pan_v3(share_url)
    except Exception as e:
        errors.append(f"方案三失败: {e}")
        print(f"   ⚠️ {e}")

    raise Exception(f"所有解析方案均失败:\n" + "\n".join(errors))


def download_file(url, filename):
    """
    下载文件并显示进度
    """
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / 1024 / 1024
        print(f"✅ {filename} 已存在（{file_size:.1f} MB），跳过下载")
        return True

    print(f"📥 正在下载 {filename}（约 509 MB），请耐心等待...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.123pan.com/',
    }

    response = requests.get(url, headers=headers, stream=True, timeout=600)
    total_size = int(response.headers.get('content-length', 0))

    if total_size == 0:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ 下载完成！")
        return True

    downloaded = 0
    chunk_size = 8192
    last_progress = -1

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                progress = int((downloaded / total_size) * 100)
                if progress % 5 == 0 and progress != last_progress:
                    print(f"  进度: {progress}%", end='\r')
                    last_progress = progress
                sys.stdout.flush()

    print(f"\n✅ 下载完成！文件大小: {os.path.getsize(filename) / 1024 / 1024:.1f} MB")
    return True


def main():
    SHARE_URL = "https://4001586487.share.123pan.cn/123pan/WGxcMh-sqqw3"

    print("📌 123云盘模型下载工具（多重解析版）")
    print(f"   分享链接: {SHARE_URL}")

    try:
        direct_url, filename = get_direct_url(SHARE_URL)
        print(f"✅ 直链获取成功")
        success = download_file(direct_url, filename)

        if success:
            print("\n🎉 模型文件下载成功！")
            print(f"📂 文件位置: {os.path.abspath(filename)}")
        else:
            print("\n❌ 下载失败")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 提示:")
        print("  1. 请检查分享链接是否正确")
        print("  2. 确保网络连接正常")
        print("  3. 如持续失败，可在浏览器中手动下载：")
        print(f"     {SHARE_URL}")
        print("  4. 手动下载后，将 rsf_model.pkl 放在本脚本同目录即可")
        sys.exit(1)


if __name__ == "__main__":
    main()
