"""Deplace le premier bloc \\begin{figure}...\\end{figure} de 03_method.tex
juste apres la ligne \\label{sec:method:overview}."""
import io

p = 'paper/sections/03_method.tex'
lines = io.open(p, encoding='utf-8').read().split('\n')

start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(r'\begin{figure}'))
end = next(i for i in range(start, len(lines)) if lines[i].strip() == r'\end{figure}')
block = lines[start:end + 1]

# retire le bloc + une ligne blanche qui suit (si presente) pour ne pas laisser de double blanc
cut_end = end + 1
if cut_end < len(lines) and lines[cut_end].strip() == '':
    cut_end += 1
removed = lines[:start] + lines[cut_end:]

ins = next(i for i, l in enumerate(removed) if 'label{sec:method:overview}' in l)
new = removed[:ins + 1] + [''] + block + [''] + removed[ins + 1:]

io.open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(new))
print(f'Figure deplacee : anciennes lignes {start+1}-{end+1} -> apres \\label{{sec:method:overview}}')
