"""登录校验（演示用）。账号任意，默认密码 123456。

代码里只保存密码的 PBKDF2-SHA256 哈希，不保存明文。要改密码：
  python -c "import hashlib;print(hashlib.pbkdf2_hmac('sha256', '新密码'.encode(), b'kdsj-2026-login', 200000).hex())"
把输出写进 用户目录/kdsj_data/password.txt（一行），程序会优先使用这个文件里的哈希。
注意：这只能防止随手打开，不是真正的安全措施。
"""
import hashlib
import hmac
from .store import data_dir

SALT = b'kdsj-2026-login'
ITER = 200000
DEFAULT_HASH = 'b7672076c12782d18e0294b8fc42e09b5a8d866008ad4a24356d648e4ad3f0a4'   # 123456


def _expected():
    p = data_dir() / 'password.txt'
    if p.exists():
        h = p.read_text(encoding='utf-8').strip()
        if len(h) == 64:
            return h
    return DEFAULT_HASH


def check(username, password):
    h = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), SALT, ITER).hex()
    return hmac.compare_digest(h, _expected())
