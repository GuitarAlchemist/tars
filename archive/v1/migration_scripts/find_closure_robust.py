import os

def find_closure_robust():
    root_dir = 'v1/src'
    # Map filename -> fullpath
    project_paths = {}
    # Map filename -> list of dependency filenames
    project_deps = {}

    # 1. Scan all projects, excluding Backup
    for dirpath, _, filenames in os.walk(root_dir):
        if 'Backup' in dirpath:
            continue
            
        for filename in filenames:
            if filename.endswith('.fsproj') or filename.endswith('.csproj'):
                fullpath = os.path.join(dirpath, filename)
                project_paths[filename] = fullpath
                deps = []
                try:
                    with open(fullpath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if 'ProjectReference' in line and 'Include="' in line:
                                # Extract path from Include="..."
                                start = line.find('Include="') + 9
                                end = line.find('"', start)
                                path = line[start:end]
                                dep_name = os.path.basename(path.replace('\\', '/'))
                                deps.append(dep_name)
                except Exception as e:
                    print(f"Error reading {fullpath}: {e}")
                project_deps[filename] = deps

    # 2. Find Closure
    target = 'TarsEngine.FSharp.Core.fsproj'
    
    if target not in project_paths:
        print(f"Target {target} not found!")
        return

    closure = set()
    stack = [target]
    
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        
        if current in project_deps:
            for dep in project_deps[current]:
                if dep not in closure:
                    stack.append(dep)
        else:
            print(f"Warning: Dependencies for {current} not known")

    print(f"Dependency Closure for {target} ({len(closure)} projects):")
    for p in closure:
        print(p)

    # Save list of kept projects
    with open('kept_projects.txt', 'w') as f:
        for p in closure:
            if p in project_paths:
                f.write(project_paths[p] + '\n')
            else:
                 print(f"Warning: Path not found for {p}")

if __name__ == '__main__':
    find_closure_robust()
