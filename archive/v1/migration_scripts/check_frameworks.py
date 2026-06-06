import os
import xml.etree.ElementTree as ET

def check_frameworks():
    root_dir = 'v1/src'
    frameworks = {} # framework -> list of projects

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.fsproj') or filename.endswith('.csproj'):
                fullpath = os.path.join(dirpath, filename)
                try:
                    with open(fullpath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple string search for TargetFramework to avoid XML namespace issues or complex parsing
                    # But XML parsing is safer if well-formed.
                    # Let's try simple parsing first.
                    try:
                        tree = ET.parse(fullpath)
                        root = tree.getroot()
                        tf = root.find('.//TargetFramework')
                        if tf is not None:
                            fw = tf.text
                        else:
                            fw = "Unknown"
                    except:
                        # Fallback to simple search
                        if '<TargetFramework>' in content:
                            start = content.find('<TargetFramework>') + len('<TargetFramework>')
                            end = content.find('</TargetFramework>', start)
                            fw = content[start:end]
                        else:
                            fw = "Not Found"
                    
                    if fw not in frameworks:
                        frameworks[fw] = []
                    frameworks[fw].append(filename)
                    
                except Exception as e:
                    print(f"Error reading {fullpath}: {e}")

    for fw, projs in frameworks.items():
        print(f"Framework: {fw} ({len(projs)} projects)")
        # print first 5
        for p in projs[:5]:
            print(f" - {p}")
        if len(projs) > 5:
            print(" ...")

if __name__ == '__main__':
    check_frameworks()
