import os
import shutil

def move_legacy():
    # Load kept projects
    with open('kept_projects.txt', 'r') as f:
        kept = set(os.path.normpath(line.strip()) for line in f if line.strip())

    # Add Tars.slnx to kept
    kept.add(os.path.normpath(os.path.abspath('v1/src/Tars.slnx')))

    root_dir = os.path.abspath('v1/src')
    dest_dir = os.path.abspath('v1/parked_legacy')
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Identify top-level folders in v1/src
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        
        # Skip if it's a file (unless it's a project file we want to move, but we usually move folders)
        # Let's look for projects inside this folder
        
        should_keep = False
        if os.path.isdir(item_path):
            # Check if any kept project is inside this folder
            for k in kept:
                if item_path in k: # Simple check if folder path is part of project path
                    should_keep = True
                    break
        else:
            # It's a file
            if item_path in kept:
                should_keep = True
            elif item.endswith('.slnx'): # Keep solution files
                should_keep = True
            elif item.endswith('.ps1') or item.endswith('.py'): # Keep scripts
                should_keep = True

        if not should_keep:
            print(f"Moving {item} to parked_legacy...")
            try:
                shutil.move(item_path, os.path.join(dest_dir, item))
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == '__main__':
    move_legacy()
