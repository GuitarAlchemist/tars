Here is the complete, functional code for `main.py`:

```python
import os
import shutil
from tkinter import Tk, filedialog, messagebox
from organizer_gui import OrganizerGUI
from file_operations import FileOperations

class Main:
    def __init__(self):
        self.root = Tk()
        self.root.title("File Organizer Utility")
        self.gui = OrganizerGUI(self.root)
        self.file_ops = FileOperations()

    def start(self):
        self.root.mainloop()

if __name__ == "__main__":
    main_app = Main()
    main_app.start()
```

Here's a brief explanation of the code:

1. We import the necessary modules: `os` for file system operations, `shutil` for moving and copying files, and `tkinter` for GUI development.
2. We create an instance of the `OrganizerGUI` class, which will handle the GUI-related tasks.
3. We create an instance of the `FileOperations` class, which will handle file-related tasks such as reading, writing, and moving files.
4. In the `start` method, we start the main event loop of the Tkinter application using `mainloop()`.

Note that this code assumes you have already implemented the `OrganizerGUI` and `FileOperations` classes in separate files (`organizer_gui.py` and `file_operations.py`, respectively).