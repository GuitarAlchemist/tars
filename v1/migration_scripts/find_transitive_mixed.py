import os
import xml.etree.ElementTree as ET

def find_transitive_mixed():
    # Load parked C# projects (filenames)
    with open('parked_list.txt', 'r') as f:
        parked_csharp = set(line.strip() for line in f if line.strip())

    root_dir = 'v1/src'
    projects = {} # path -> list of dependencies (filenames)
    project_paths = {} # filename -> fullpath

    # 1. Build Dependency Graph
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.fsproj'):
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
                projects[filename] = deps

    # 2. Identify Bad Projects (Direct C# deps)
    bad_projects = set()
    for p, deps in projects.items():
        for d in deps:
            if d in parked_csharp:
                bad_projects.add(p)
                print(f"Direct dependency: {p} -> {d}")
                break

    # 3. Propagate Badness (Transitive deps)
    changed = True
    while changed:
        changed = False
        for p, deps in projects.items():
            if p in bad_projects:
                continue
            for d in deps:
                if d in bad_projects:
                    bad_projects.add(p)
                    print(f"Transitive dependency: {p} -> {d}")
                    changed = True
                    break

    # 4. Output List
    with open('mixed_projects_transitive.txt', 'w') as f:
        for p in bad_projects:
            if p in project_paths:
                f.write(project_paths[p] + '\n')
            else:
                print(f"Warning: Path not found for {p}")

    print(f"Found {len(bad_projects)} F# projects with transitive dependencies on C#.")

if __name__ == '__main__':
    find_transitive_mixed()
