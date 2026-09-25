"""Check test results contents per marker."""
from pathlib import Path
test_dir = Path(r'E:\aic\final-ihc\results_v3\test')
for m in ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']:
    d = test_dir / m
    files = list(d.glob('*.png')) + list(d.glob('*.jpg'))
    print(f'{m}: {len(files)} files')
print('total files:', sum(len(list((test_dir/m).glob("*"))) for m in ['HLA-DR','CD68','CD45RO','Vimentin']))
