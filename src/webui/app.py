"""webui/app.py — Flask 应用入口

启动: python -m src.webui.app
访问: http://127.0.0.1:5000
"""

import sys
import os
from pathlib import Path

# 确保项目根目录在 path 中
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ⚠️ 必须在 import 任何 akshare/requests 之前禁用系统代理（东财接口依赖）
from src.webui.proxy import disable_proxy_globally
disable_proxy_globally()

# 修复 Windows GBK 终端下中文输出乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, send_from_directory

from src.webui.api import api


def create_app() -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False  # 中文不被 \u 转义

    app.register_blueprint(api, url_prefix="/api")

    static_dir = Path(__file__).resolve().parent / "static"

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(static_dir, filename)

    return app


app = create_app()


if __name__ == "__main__":
    print("📊 Stock Advisor Web UI")
    print("   http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
