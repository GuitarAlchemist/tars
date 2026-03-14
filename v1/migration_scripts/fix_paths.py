import os

def fix_paths():
    root_dir = 'v1/src'
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.fsproj'):
                fullpath = os.path.join(dirpath, filename)
                try:
                    with open(fullpath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Replace ..\src\ with ..\
                    # Use double backslashes for literal backslash
                    new_content = content.replace('..\\src\\', '..\\')
                    new_content = new_content.replace('../src/', '../')
                    
                    if new_content != content:
                        print(f"Fixing paths in {filename}")
                        with open(fullpath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                except Exception as e:
                    print(f"Error processing {fullpath}: {e}")

if __name__ == '__main__':
    fix_paths()
