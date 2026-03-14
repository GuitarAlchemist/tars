import os

def fast_find():
    targets = {
        'TARS.AI.Inference.fsproj',
        'Tars.Engine.Grammar.fsproj',
        'Tars.Engine.VectorStore.fsproj'
    }
    found = {}
    
    for dirpath, _, filenames in os.walk('v1'):
        for filename in filenames:
            if filename in targets:
                found[filename] = os.path.join(dirpath, filename)
    
    for filename, path in found.items():
        print(f"Found {filename}: {path}")
        
    # Append to kept_projects.txt
    with open('kept_projects.txt', 'a') as f:
        for path in found.values():
            f.write(path + '\n')

if __name__ == '__main__':
    fast_find()
