from pathlib import Path


intermediate_file_extensions = [
    '*.aux',
    '*.bbl',
    '*.blg',
    '*.brf',
    '*.log',
    '*.out',
    '*.Pages',
    '*.toc',
    '**/*.aux',
    '*.lof',
    '*.lot',
    '*.lod',
    '*.acn',
    '*.acr',
    '*.alg',
    '*.glg',
    '*.glo',
    '*.gls',
    '*.ist',
    '*.mlg',
    '*.mni',
    '*.mno',
    '*.loa',
    '*.lol',
    '*.sync*',
    'thesis.pdf',
    '**/*-eps-converted-to.pdf',
    'output/'
]


def remove_recursive(path):
    if path.is_dir():
        entries = list(path.iterdir())
        if len(entries) == 0:
            path.rmdir()
            return
        for entry in path.iterdir():
            remove_recursive(entry)
    else:
        path.unlink()


if __name__ == '__main__':
    for extension in intermediate_file_extensions:
        for path in Path('.').glob(extension):
            remove_recursive(path)
