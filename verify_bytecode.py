"""Compile gpu_runtime.py and compare per-function bytecode (opname, argval) with the original pyc.
Line numbers, co_filename and memory addresses are ignored. Run: python3 verify_bytecode.py [big.pyc] [gpu_runtime.py]"""
import dis, hashlib, marshal, sys, types
here = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
pyc = sys.argv[1] if len(sys.argv) > 1 else here + '/ref/gpu_runtime_original.cpython-311.pyc'
src = sys.argv[2] if len(sys.argv) > 2 else here + '/gpu_runtime.py'
raw = open(pyc, 'rb').read()
print('python', sys.version.split()[0], 'pyc magic', raw[:4].hex(), 'bytes', len(raw))
print('big.pyc sha256', hashlib.sha256(raw).hexdigest())
print('gpu_runtime.py sha256', hashlib.sha256(open(src, 'rb').read()).hexdigest())
orig = marshal.loads(raw[16:])
new = compile(open(src, encoding='utf-8').read(), 'gpu_runtime.py', 'exec')

def norm(v):
    if isinstance(v, types.CodeType):
        return ('<code>', v.co_name)
    return v

def ins(c):
    return [(i.opname, norm(i.argval)) for i in dis.get_instructions(c)]

def codes(c, prefix=''):
    out = {prefix + c.co_name: c}
    n = 0
    for k in c.co_consts:
        if isinstance(k, types.CodeType):
            n += 1
            sub = codes(k, prefix + c.co_name + '.')
            if k.co_name == '<dictcomp>':  # same name several times: number by order of appearance
                sub = {key.replace('<dictcomp>', '<dictcomp#%d>' % n, 1): v for key, v in sub.items()}
            out.update(sub)
    return out

a, b = codes(orig), codes(new)
bad = 0
for n in sorted(set(a) | set(b)):
    if n not in a or n not in b:
        print('MISSING', n, 'in', 'recon' if n in a else 'orig'); bad += 1; continue
    x, y = ins(a[n]), ins(b[n])
    meta = [(f, getattr(a[n], f) == getattr(b[n], f)) for f in ('co_varnames', 'co_names', 'co_argcount', 'co_flags')]
    extra = [f for f, ok in meta if not ok]
    if x == y and not extra:
        print('OK      ', n, len(x), 'instructions')
    else:
        bad += 1
        print('MISMATCH', n, len(x), 'vs', len(y), 'meta differs:', extra)
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q:
                print('   first diff at', i, p, '!=', q); break
print('defaults orig/recon vmd:', [k for k in orig.co_consts if isinstance(k, tuple)][-1])
print('RESULT', 'ALL IDENTICAL' if not bad else f'{bad} MISMATCH(ES)')
sys.exit(1 if bad else 0)
