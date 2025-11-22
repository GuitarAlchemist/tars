import os

def list_projects():
    root_dir = 'v1/src'
    projects = []
    
    # Exclude these directories
    excludes = {'parked_legacy', 'parked_csharp', 'Backup'}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter out excluded directories
        # Modify dirnames in-place to prevent recursion into excluded dirs
        dirnames[:] = [d for d in dirnames if d not in excludes and not any(e in os.path.join(dirpath, d) for e in excludes)]
        
        for filename in filenames:
            if filename.endswith('.fsproj') or filename.endswith('.csproj'):
                # Get path relative to v1/src
                fullpath = os.path.join(dirpath, filename)
                relpath = os.path.relpath(fullpath, root_dir)
                projects.append(relpath)

    with open('project_list.txt', 'w', encoding='utf-8-sig') as f:
        for p in projects:
            f.write(p + '\n')
            
    print(f"Found {len(projects)} projects.")
    for p in projects:
        print(p)

if __name__ == '__main__':
    list_projects()
