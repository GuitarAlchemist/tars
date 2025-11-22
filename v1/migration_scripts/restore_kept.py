import os
import shutil

def restore_kept():
    targets = [
        'TarsEngine.FSharp.Core',
        'TarsEngine.FSharp.Metascript',
        'TARS.AI.Inference',
        'Tars.Engine.Grammar',
        'Tars.Engine.VectorStore'
    ]
    
    src_base = 'v1/parked_legacy'
    dest_base = 'v1/src'
    
    if not os.path.exists(dest_base):
        os.makedirs(dest_base)

    # Find where they are
    for root, dirs, files in os.walk(src_base):
        for d in dirs:
            if d in targets:
                src_path = os.path.join(root, d)
                dest_path = os.path.join(dest_base, d)
                
                print(f"Restoring {d} from {src_path} to {dest_path}...")
                if os.path.exists(dest_path):
                    print(f"  Destination {dest_path} already exists. Skipping or merging?")
                    # If it exists, maybe it's empty or partial?
                    # Let's assume if it exists in src, we don't need to restore, 
                    # UNLESS the one in legacy is the 'real' one and src is empty.
                    # For safety, let's not overwrite if exists, but warn.
                else:
                    try:
                        shutil.move(src_path, dest_path)
                        print("  Restored.")
                    except Exception as e:
                        print(f"  Failed to restore: {e}")

if __name__ == '__main__':
    restore_kept()
