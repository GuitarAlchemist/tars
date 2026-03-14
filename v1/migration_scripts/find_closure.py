import os
import xml.etree.ElementTree as ET

def find_closure():
    root_dir = 'v1/src'
    # Map filename -> fullpath
    project_paths = {}
    # Map filename -> list of dependency filenames
    project_deps = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.fsproj') or filename.endswith('.csproj'):
                fullpath = os.path.join(dirpath, filename)
                project_paths[filename] = fullpath
                deps = []
                try:
                    tree = ET.parse(fullpath)
                    root = tree.getroot()
                    for item_group in root.findall('ItemGroup'):
                        for proj_ref in item_group.findall('ProjectReference'):
                            include = proj_ref.get('Include')
                            if include:
                                dep_name = os.path.basename(include.replace('\\', '/'))
                                deps.append(dep_name)
                except Exception as e:
                    print(f"Error parsing {fullpath}: {e}")
                project_deps[filename] = deps

    # Target project
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
                    pass # Already processed
        else:
            print(f"Warning: Dependencies for {current} not known (maybe external or missing)")

    print(f"Dependency Closure for {target} ({len(closure)} projects):")
    for p in closure:
        print(p)

    # Save list of kept projects
    with open('kept_projects.txt', 'w') as f:
        for p in closure:
            if p in project_paths:
                f.write(project_paths[p] + '\n')

if __name__ == '__main__':
    find_closure()
