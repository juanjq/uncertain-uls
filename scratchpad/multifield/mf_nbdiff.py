import json, sys
def outs(path):
    d = json.load(open(path)); res = []
    for i, c in enumerate(d["cells"]):
        if c["cell_type"] != "code": continue
        t = []
        for o in c.get("outputs", []):
            if o["output_type"] == "stream": t.append("".join(o["text"]))
            elif o["output_type"] in ("execute_result", "display_data") and "text/plain" in o["data"] and "image/png" not in o["data"]:
                t.append("".join(o["data"]["text/plain"]))
            elif o["output_type"] == "error": t.append("ERROR " + o["ename"])
        res.append((i, "\n".join(x for x in t if not x.startswith("[cache]"))))
    return res
a, b = outs(sys.argv[1]), outs(sys.argv[2])
for (i, x), (_, y) in zip(a, b):
    xs = "\n".join(l for l in x.splitlines() if not l.startswith("[cache]"))
    ys = "\n".join(l for l in y.splitlines() if not l.startswith("[cache]"))
    if xs != ys:
        print(f"--- cell {i} BEFORE\n{xs[:1500]}\n--- AFTER\n{ys[:1500]}\n")
