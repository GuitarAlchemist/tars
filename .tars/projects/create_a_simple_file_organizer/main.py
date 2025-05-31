Here is the complete, working content for the `main.py` file:

```python
import tkinter as tk
from tkinter import filedialog
from organizer_gui import OrganizerGUI
from file_organizer import FileOrganizer
import os
import shutil

class MainApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("File Organizer")
        self.organizer_gui = OrganizerGUI(self)
        self.file_organizer = FileOrganizer()

    def start(self):
        self.organizer_gui.pack(fill="both", expand=True)
        self.root.mainloop()

def main():
    app = MainApp()
    app.start()

if __name__ == "__main__":
    main()
```

This code sets up a Tkinter GUI application with two main components: `OrganizerGUI` and `FileOrganizer`. The `OrganizerGUI` class is responsible for creating the GUI elements, such as buttons and text fields. The `FileOrganizer` class handles the file organization logic.

The `MainApp` class serves as the entry point of the application, initializing the GUI and starting the main event loop.

Note that this code assumes you have already implemented the `organizer_gui.py` and `file_organizer.py` files according to the analysis provided earlier.