"""Epingle (\\begin{table/figure}[tbp] -> [H]) les flottants de Main Results,
c.-a-d. tout ce qui precede \\subsection{Analysis}, pour qu'ils ne derivent plus
dans la section Analysis."""
import io

p = 'paper/sections/04_experiments.tex'
lines = io.open(p, encoding='utf-8').read().split('\n')
cut = next(i for i, l in enumerate(lines) if r'\subsection{Analysis}' in l)

n = 0
for i in range(cut):
    for env in ('table', 'figure'):
        src = '\\begin{%s}[tbp]' % env
        if src in lines[i]:
            lines[i] = lines[i].replace(src, '\\begin{%s}[H]' % env)
            n += 1

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
print(f'{n} flottants Main Results epingles en [H] (avant \\subsection{{Analysis}} ligne {cut+1})')
