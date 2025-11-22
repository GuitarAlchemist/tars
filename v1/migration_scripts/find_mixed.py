import os
import xml.etree.ElementTree as ET

def find_mixed_projects():
    # Load parked C# projects
    with open('parked_list.txt', 'r') as f:
        parked_csharp = set(line.strip() for line in f if line.strip())

    root_dir = 'v1/src'
    mixed_projects = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.fsproj'):
                fullpath = os.path.join(dirpath, filename)
                try:
                    tree = ET.parse(fullpath)
                    root = tree.getroot()
                    
                    has_ref = False
                    for item_group in root.findall('ItemGroup'):
                        for proj_ref in item_group.findall('ProjectReference'):
                            include = proj_ref.get('Include')
                            if include:
                                # Check if the referenced file is in our parked list
                                ref_name = os.path.basename(include.replace('\\', '/'))
                                if ref_name in parked_csharp:
                                    has_ref = True
                                    print(f"Project {filename} references parked {ref_name}")
                                    break
                        if has_ref:
                            break
                    
                    if has_ref:
                        mixed_projects.append(fullpath)
                except Exception as e:
                    print(f"Error parsing {fullpath}: {e}")

    with open('mixed_projects.txt', 'w') as f:
        for p in mixed_projects:
            f.write(p + '\n')
            
    print(f"Found {len(mixed_projects)} F# projects with dependencies on parked C# projects.")

if __name__ == '__main__':
    find_mixed_projects()
