import os

def list_duplicates():
    root_dir = 'v1/src'
    target = 'TarsEngine.FSharp.Core.fsproj'
    for dirpath, _, filenames in os.walk(root_dir):
        if target in filenames:
            print(os.path.join(dirpath, target))

if __name__ == '__main__':
    list_duplicates()
