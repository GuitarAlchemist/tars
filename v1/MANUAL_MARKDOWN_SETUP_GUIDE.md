# Manual Setup Guide: Configure Mark Text for Markdown Files

**Purpose:** Set up Mark Text as the default viewer for all Markdown files in Windows

---

## 🎯 Quick Setup (Automated)

### Option 1: Run the Batch Script (Recommended)
1. **Right-click** on `setup_markdown_associations.bat`
2. **Select** "Run as administrator"
3. **Follow** the on-screen instructions
4. **Test** by double-clicking any `.md` file

### Option 2: Run the PowerShell Script
1. **Right-click** on PowerShell
2. **Select** "Run as administrator"
3. **Navigate** to the TARS directory
4. **Run:** `.\setup_markdown_file_associations.ps1`

---

## 🔧 Manual Setup (If Scripts Don't Work)

### Step 1: Find Mark Text Installation Path

Check these locations for `Mark Text.exe`:
- `%LOCALAPPDATA%\Programs\marktext\Mark Text.exe`
- `%ProgramFiles%\Mark Text\Mark Text.exe`
- `%ProgramFiles(x86)%\Mark Text\Mark Text.exe`

### Step 2: Configure File Associations

#### Method A: Using Windows Settings
1. **Right-click** any `.md` file
2. **Select** "Open with" → "Choose another app"
3. **Check** "Always use this app to open .md files"
4. **Browse** to Mark Text installation path
5. **Select** `Mark Text.exe`
6. **Click** "OK"

#### Method B: Using Registry Editor (Advanced)
1. **Press** `Win + R`, type `regedit`, press Enter
2. **Navigate** to `HKEY_CLASSES_ROOT\.md`
3. **Set** default value to `MarkText.md`
4. **Create** key `HKEY_CLASSES_ROOT\MarkText.md`
5. **Set** default value to "Markdown Document"
6. **Create** subkey `shell\open\command`
7. **Set** default value to `"C:\Path\To\Mark Text.exe" "%1"`

### Step 3: Add Context Menu (Optional)
1. **Navigate** to `HKEY_CLASSES_ROOT\*\shell`
2. **Create** new key named `MarkText`
3. **Set** default value to "Open with Mark Text"
4. **Create** subkey `command`
5. **Set** default value to `"C:\Path\To\Mark Text.exe" "%1"`

---

## 🧪 Testing Your Setup

### Test Files to Open:
```
C:\Users\spare\source\repos\tars\TARS_Comprehensive_Documentation\TARS_Executive_Summary_Comprehensive.md
C:\Users\spare\source\repos\tars\TARS_Comprehensive_Documentation\TARS_Technical_Specification_Comprehensive.md
C:\Users\spare\source\repos\tars\TARS_Comprehensive_Documentation\TARS_API_Documentation.md
```

### What You Should See:
- ✅ **Mermaid diagrams** rendered as beautiful graphics
- ✅ **Mathematical formulas** displayed properly
- ✅ **Professional formatting** with syntax highlighting
- ✅ **Live preview** as you scroll through the document

---

## 🎨 Mark Text Features for TARS Documentation

### Mermaid Diagram Support
- **System Architecture** diagrams
- **Flowcharts** and process flows
- **Timeline** charts
- **Pie charts** and metrics
- **Class diagrams** and UML

### Mathematical Formula Support
- **LaTeX/KaTeX** rendering
- **Neuromorphic computing** equations
- **Optical computing** formulas
- **Quantum computing** state representations
- **CUDA performance** models

### Professional Features
- **Multiple themes** (light/dark/custom)
- **Export options** (HTML, PDF)
- **Live preview** mode
- **Syntax highlighting**
- **Table editing**

---

## 🔍 Troubleshooting

### Mark Text Not Found
1. **Download** from: https://github.com/marktext/marktext/releases
2. **Install** `marktext-setup.exe`
3. **Run** setup script again

### File Associations Not Working
1. **Restart** Windows Explorer:
   - Press `Ctrl + Shift + Esc`
   - Find "Windows Explorer"
   - Right-click → "Restart"
2. **Reboot** computer if necessary

### Mermaid Diagrams Not Rendering
1. **Check** Mark Text version (should be 0.17.1+)
2. **Update** Mark Text if needed
3. **Verify** Mermaid syntax in documents

### Permission Issues
1. **Run** scripts as Administrator
2. **Check** UAC settings
3. **Verify** registry permissions

---

## 📋 Alternative Markdown Viewers

If Mark Text doesn't work, try these alternatives:

### Typora ($14.99)
- **Download:** https://typora.io/
- **Features:** Excellent Mermaid support, live preview
- **Setup:** Install and set as default in Windows Settings

### Obsidian (Free)
- **Download:** https://obsidian.md/
- **Features:** Mermaid plugin, knowledge management
- **Setup:** Install Mermaid plugin after installation

### Visual Studio Code (Free)
- **Extensions:** "Markdown Preview Enhanced"
- **Features:** Full Mermaid support, developer-friendly
- **Setup:** Install extension, use Ctrl+Shift+V for preview

---

## ✅ Verification Checklist

After setup, verify these work:

- [ ] Double-clicking `.md` files opens Mark Text
- [ ] Mermaid diagrams render properly
- [ ] Mathematical formulas display correctly
- [ ] Context menu shows "Open with Mark Text"
- [ ] TARS documentation displays beautifully
- [ ] Export functions work (HTML/PDF)

---

## 🎉 Success!

Once configured, you'll have:
- ✅ **Professional Markdown viewing** with Mark Text
- ✅ **Beautiful Mermaid diagrams** in your TARS documentation
- ✅ **Mathematical formulas** rendered perfectly
- ✅ **Easy access** via double-click or context menu
- ✅ **Export capabilities** for sharing documentation

**Your comprehensive TARS technical documentation will now display with professional quality, making it perfect for stakeholder presentations and technical reviews!**
