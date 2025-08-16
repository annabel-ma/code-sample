import sys, json, os

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    obj = json.load(f)
line = json.dumps(obj, separators=(',', ':'))  
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, 'a') as out:
    out.write(line + '\n')
