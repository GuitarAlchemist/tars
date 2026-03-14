import os

def generate_lang_slnx():
    with open('project_list.txt', 'r', encoding='utf-8') as f:
        projects = [line.strip() for line in f if line.strip()]

    projects = sorted(list(set(projects)))
    
    # Exclude known duplicates
    excludes = [
        'Legacy_CSharp_Projects\\TarsCliMinimal.csproj',
        'Legacy_CSharp_Projects\\TarsCliMinimal.Tests.csproj'
    ]
    projects = [p for p in projects if p not in excludes]

    csharp_projects = [p for p in projects if p.endswith('.csproj')]
    fsharp_projects = [p for p in projects if p.endswith('.fsproj')]

    def write_slnx(filename, projs):
        slnx_content = ['<Solution>', '  <Configurations>', '    <Platform Name="Any CPU" />', '    <Platform Name="x64" />', '    <Platform Name="x86" />', '  </Configurations>']
        for proj in projs:
            proj_path = proj.replace('\\', '/').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
            if proj.endswith('.fsproj'):
                slnx_content.append(f'  <Project Path="{proj_path}" Type="Classic F#" />')
            else:
                slnx_content.append(f'  <Project Path="{proj_path}" />')
        slnx_content.append('</Solution>')
        
        with open(f'v1/src/{filename}', 'w', encoding='utf-8') as f:
            f.write('\n'.join(slnx_content))

    write_slnx('CSharpOnly.slnx', csharp_projects)
    write_slnx('FSharpOnly.slnx', fsharp_projects)

if __name__ == '__main__':
    generate_lang_slnx()
