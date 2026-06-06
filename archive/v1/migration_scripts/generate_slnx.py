import os

def generate_slnx():
    # Use utf-8-sig to handle BOM from PowerShell Out-File
    with open('project_list.txt', 'r', encoding='utf-8-sig') as f:
        projects = [line.strip() for line in f if line.strip()]

    # Filter for F# projects only
    projects = [p for p in projects if p.endswith('.fsproj')]

    # Deduplicate by filename (project name), resolving conflicts
    project_map = {} # filename -> path
    
    # Sort projects to ensure deterministic processing
    # We want to prioritize:
    # 1. Non-Legacy folders
    # 2. Shorter paths
    
    def sort_key(p):
        is_legacy = 'Legacy' in p
        return (is_legacy, len(p), p)

    projects = sorted(list(set(projects)), key=sort_key)

    for proj in projects:
        name_without_ext = os.path.splitext(os.path.basename(proj))[0]
        key = name_without_ext
        
        if key not in project_map:
            project_map[key] = proj
        else:
            print(f"Skipping duplicate project name '{key}': {proj} (kept {project_map[key]})")

    unique_projects = sorted(project_map.values())

    slnx_content = ['<Solution>', '  <Configurations>', '    <Platform Name="Any CPU" />', '    <Platform Name="x64" />', '    <Platform Name="x86" />', '  </Configurations>']

    for proj in unique_projects:
        # Use forward slashes
        proj_path = proj.replace('\\', '/').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
        
        # Remove any potential invisible characters
        proj_path = ''.join(c for c in proj_path if c.isprintable())

        slnx_content.append(f'  <Project Path="{proj_path}" Type="Classic F#" />')

    slnx_content.append('</Solution>')

    with open('v1/src/Tars.slnx', 'w', encoding='utf-8') as f:
        f.write('\n'.join(slnx_content))

if __name__ == '__main__':
    generate_slnx()
