import os

def identify_moves():
    with open('project_list.txt', 'r', encoding='utf-8-sig') as f:
        projects = [line.strip() for line in f if line.strip()]

    csharp_projects = [p for p in projects if p.endswith('.csproj')]
    fsharp_projects = [p for p in projects if p.endswith('.fsproj')]

    # Identify directories containing C# projects
    csharp_dirs = set()
    for p in csharp_projects:
        # Get the directory of the project
        d = os.path.dirname(p)
        # If the directory is empty (root of v1/src), use the project name as the folder to move? 
        # No, if it's at root, we can't move the root.
        # But most are in subfolders.
        if d:
            csharp_dirs.add(d)
        else:
            # Root level csproj. We should move the file itself, or create a folder for it?
            # The user said "parked somewhere".
            pass

    # Identify directories containing F# projects (to avoid moving them)
    fsharp_dirs = set()
    for p in fsharp_projects:
        d = os.path.dirname(p)
        if d:
            fsharp_dirs.add(d)
            # Also add all parent directories to avoid moving a parent of an F# project
            parts = d.split('\\')
            for i in range(1, len(parts)):
                fsharp_dirs.add('\\'.join(parts[:i]))

    # Filter csharp_dirs
    # We want to move the *top-most* directory that contains ONLY C# stuff.
    # But for simplicity, let's just look at the project directories first.
    
    moves = []
    
    # Group by top-level folder in v1/src
    # e.g. Legacy_CSharp_Projects/...
    
    root_folders = set()
    for p in csharp_projects:
        parts = p.split('\\')
        if len(parts) > 1:
            root_folders.add(parts[0])
    
    # Check if these root folders contain any F# projects
    safe_root_folders = []
    for rf in root_folders:
        has_fsharp = False
        for fp in fsharp_projects:
            if fp.startswith(rf + '\\'):
                has_fsharp = True
                break
        if not has_fsharp:
            safe_root_folders.append(rf)
            
    # For projects not in safe_root_folders, we need to dig deeper.
    # Or just move the specific project folder if it doesn't contain F#
    
    # Let's just list what we found
    print("Safe Root Folders to Move:")
    for f in safe_root_folders:
        print(f)
        
    print("\nOther C# Projects (Mixed Roots):")
    for p in csharp_projects:
        parts = p.split('\\')
        if len(parts) > 1 and parts[0] not in safe_root_folders:
            print(p)
        elif len(parts) == 1:
             print(f"Root file: {p}")

if __name__ == '__main__':
    identify_moves()
