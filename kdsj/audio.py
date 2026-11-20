"""音频读取与分帧（只依赖 numpy）。"""
import struct
import numpy as np


def read_wav(src):
    """读取 wav 文件或其字节内容，返回 (采样点 x 声道 的 float64 数组, 采样率)。

    支持 PCM 8/16/24/32 位整数、32/64 位浮点，以及多声道录音常用的 WAVE_FORMAT_EXTENSIBLE
    （Python 自带的 wave 模块不支持这种格式）。整数格式按满量程缩放到 [-1, 1)。
    """
    data = src if isinstance(src, (bytes, bytearray)) else open(src, 'rb').read()
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('不是 RIFF/WAVE 格式的文件')
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b'fmt ':
            tag, ch, fs, _, _, bits = struct.unpack('<HHIIHH', body[:16])
            if tag == 0xFFFE and len(body) >= 26:      # EXTENSIBLE：实际格式在子格式 GUID 的前 2 字节
                tag = struct.unpack('<H', body[24:26])[0]
            fmt = (tag, ch, fs, bits)
        elif cid == b'data':
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError('wav 文件缺少 fmt 或 data 块')
    tag, ch, fs, bits = fmt
    if tag == 1:
        if bits == 8:
            x = np.frombuffer(raw, np.uint8).astype(np.float64) / 128 - 1
        elif bits == 16:
            x = np.frombuffer(raw[:len(raw) // 2 * 2], '<i2') / 32768.0
        elif bits == 24:
            b = np.frombuffer(raw[:len(raw) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
        elif bits == 32:
            x = np.frombuffer(raw[:len(raw) // 4 * 4], '<i4') / float(1 << 31)
        else:
            raise ValueError(f'不支持 {bits} 位整数 wav')
    elif tag == 3:
        dt = '<f4' if bits == 32 else '<f8'
        x = np.frombuffer(raw[:len(raw) // (bits // 8) * (bits // 8)], dt).astype(np.float64)
    else:
        raise ValueError(f'不支持的 wav 编码（格式号 {tag}）')
    n = len(x) // ch
    return x[:n * ch].reshape(n, ch), fs


def write_wav(path, x, fs):
    """把 [-1, 1] 范围的单声道或多声道信号写成 16 位 PCM wav。"""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1, 1 - 1 / 32768) * 32768).astype('<i2')
    ch = x.shape[1]
    body = pcm.tobytes()
    hdr = b'RIFF' + struct.pack('<I', 36 + len(body)) + b'WAVE'
    hdr += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, ch, fs, fs * ch * 2, ch * 2, 16)
    hdr += b'data' + struct.pack('<I', len(body))
    with open(path, 'wb') as f:
        f.write(hdr + body)


def load_signal(path, channel=0):
    """读取单通道信号。支持 .wav、.npy、.csv/.txt（取第 channel 列）。返回 (信号, 采样率或 None)。"""
    p = str(path).lower()
    if p.endswith('.wav'):
        X, fs = read_wav(path)
        return X[:, min(channel, X.shape[1] - 1)], fs
    if p.endswith('.npy'):
        x = np.load(path)
        return (x if x.ndim == 1 else x[:, channel]).astype(np.float64), None
    if p.endswith('.csv') or p.endswith('.txt'):
        x = np.loadtxt(path, delimiter=',' if p.endswith('.csv') else None, ndmin=2)
        return x[:, min(channel, x.shape[1] - 1)], None
    raise ValueError(f'不支持的文件类型：{path}')


def frames(x, win, hop):
    """把信号切成长度 win、步长 hop 的窗口，返回 (窗口数 x win) 和每个窗口的起点。不足一个窗口时补零。"""
    x = np.asarray(x, np.float64).ravel()
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    starts = np.arange(0, x.size - win + 1, hop)
    return np.stack([x[s:s + win] for s in starts]), starts
