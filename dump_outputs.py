"""Print text outputs of the executed notebook (for review); flags errors/tracebacks/warnings."""
import sys
import nbformat

nb = nbformat.read(sys.argv[1] if len(sys.argv) > 1 else 'dz_topic_4_Джафарлі_Фарід.ipynb', as_version=4)
n_err = n_img = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    print(f'\n===== cell {i} =====')
    print(c.source.splitlines()[0][:100])
    print('-----')
    for o in c.get('outputs', []):
        if o.output_type == 'error':
            n_err += 1
            print('ERROR:', o.ename, o.evalue)
        elif o.output_type == 'stream':
            if o.name == 'stderr':
                print('[stderr]', o.text[:2000])
            else:
                print(o.text[:4000])
        else:
            data = o.get('data', {})
            if 'image/png' in data:
                n_img += 1
                print('[image]')
            elif 'text/plain' in data:
                print(data['text/plain'][:4000])
print(f'\nerrors: {n_err}, images: {n_img}')
