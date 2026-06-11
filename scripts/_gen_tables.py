"""Genere les corps de tables LaTeX depuis outputs/results/aggregate/results_all.json."""
import json

R = json.load(open('outputs/results/aggregate/results_all.json'))['results']
cols = ['crossner_ai','crossner_literature','crossner_music','crossner_politics',
        'crossner_science','wnut17','mit_restaurant','mit_movie','fabner',
        'bionlp2004','conll2003','gum','gentle']
comp = ['GLiNER-S','GLiNER-M','GLiNER-L','GNER-T5-base','Opener-V2-e2e']  # comparables e2e (gras)
disp = {'GLiNER-S':'GLiNER-S','GLiNER-M':'GLiNER-M','GLiNER-L':'GLiNER-L',
        'GNER-T5-base':'GNER','OWNER-e2e':r'OWNER$^\dagger$','Opener-V2-e2e':'Opener',
        'OWNER':r'OWNER$^\dagger$','Opener-V2-gold':'Opener'}
BF = r'\textbf{'
EOL = r' \\'

def val(m, ds, k):
    c = R.get(m, {}).get(ds)
    return c.get(k) if c else None

def avg(m, k):
    vs = [val(m, ds, k) for ds in cols if val(m, ds, k) is not None]
    return sum(vs)/len(vs) if vs else None

def complete(m):
    return all(val(m, ds, 'ami') is not None for ds in cols)

def best(k, better):
    b = {}
    for ds in cols + ['avg']:
        vals = [(avg(mm, k) if ds == 'avg' else val(mm, ds, k), mm) for mm in comp]
        vals = [(v, mm) for v, mm in vals if v is not None]
        if vals:
            b[ds] = better(vals)[1]
    return b

def gen(rows, k, scale, fmt, better):
    bp = best(k, better)
    out = []
    for m in rows:
        cells = []
        for ds in cols:
            v = val(m, ds, k)
            if v is None:
                cells.append('--'); continue
            s = fmt(v*scale)
            if bp.get(ds) == m:
                s = BF + s + '}'
            cells.append(s)
        a = avg(m, k)
        if a is None:
            ac = '--'
        else:
            ac = fmt(a*scale)
            if bp.get('avg') == m and complete(m):
                ac = BF + ac + '}'
            if not complete(m):
                ac = ac + r'$^\dagger$'
        out.append(disp[m] + ' & ' + ' & '.join(cells) + ' & ' + ac + EOL)
    return '\n'.join(out)

f1 = lambda x: f'{x:.1f}'
f0 = lambda x: f'{x:.0f}'
f2 = lambda x: f'{x:.2f}'
mx, mn = max, min
e2e = ['GLiNER-S','GLiNER-M','GLiNER-L','GNER-T5-base','OWNER-e2e','Opener-V2-e2e']
gold = ['OWNER','Opener-V2-gold']

print('% ===== TABLE 2 AMI x100 : end-to-end block =====')
print(gen(e2e, 'ami', 100, f1, mx))
print('% ----- typing-on-gold block -----')
print(gen(gold, 'ami', 100, f1, mx))
print('\n% ===== TABLE 3 latency p50 (ms) =====')
print(gen(e2e, 'p50_ms', 1, f0, mn))
print('\n% ===== TABLE 4 energy (Wh) =====')
print(gen(e2e, 'kwh', 1000, f2, mn))

# ---- summary ----
params = {'GLiNER-S':'50M','GLiNER-M':'200M','GLiNER-L':'330M','GNER-T5-base':'220M',
          'OWNER-e2e':'110M','Opener-V2-e2e':'137M'}
print('\n% ===== TABLE 5 summary (Params | AMI x100 | p50 ms | Energy Wh | CO2 g) =====')
for m in e2e:
    ami = avg(m,'ami'); p = avg(m,'p50_ms'); wh = avg(m,'kwh'); co = avg(m,'gco2eq')
    def g(x, fmt): return fmt(x) if x is not None else '--'
    dag = '' if complete(m) else r'$^\dagger$'
    row = f"{disp[m]} & {params[m]} & {g(ami, lambda x: f'{x*100:.1f}')}{dag} & {g(p, f0)} & {g(wh, lambda x: f'{x*1000:.2f}')} & {g(co, f2)}" + EOL
    print(row)
