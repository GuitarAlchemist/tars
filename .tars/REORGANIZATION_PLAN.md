# 🗂️ TARS Directory Reorganization Plan

## 🎯 **PROPOSED NEW STRUCTURE**

### **Option A: "system" subfolder (Recommended)**
```
.tars/
├── tars.yaml                    # Main TARS configuration (stays at root)
├── ORGANIZATION_SUMMARY.md      # Project overview (stays at root)
│
├── system/                      # 🔧 INTERNAL TARS SYSTEM RESOURCES
│   ├── config/                  # TARS internal configuration
│   ├── metascripts/             # Core TARS metascripts
│   ├── templates/               # System templates
│   ├── workflows/               # Internal workflows
│   ├── scripts/                 # System utility scripts
│   └── knowledge/               # TARS knowledge base
│
├── workspace/                   # 👤 USER WORKSPACE
│   ├── docs/                    # User documentation
│   ├── examples/                # User examples
│   ├── explorations/            # User research/explorations
│   ├── plans/                   # User planning (TODOs, etc.)
│   └── projects/                # User project files
│
└── shared/                      # 🤝 SHARED RESOURCES
    ├── templates/               # Shared templates
    ├── libraries/               # Shared code libraries
    └── assets/                  # Shared assets
```

### **Option B: "core" subfolder (Alternative)**
```
.tars/
├── tars.yaml                    # Main TARS configuration
├── ORGANIZATION_SUMMARY.md      # Project overview
│
├── core/                        # 🔧 TARS CORE SYSTEM
│   ├── engine/                  # Core engine components
│   ├── metascripts/             # System metascripts
│   ├── config/                  # System configuration
│   ├── workflows/               # Core workflows
│   └── intelligence/            # Intelligence system
│
├── user/                        # 👤 USER SPACE
│   ├── docs/                    # User documentation
│   ├── projects/                # User projects
│   ├── explorations/            # User explorations
│   └── plans/                   # User plans
│
└── shared/                      # 🤝 SHARED RESOURCES
    ├── templates/               # Shared templates
    └── libraries/               # Shared libraries
```

## 🎯 **RECOMMENDATION: Option A ("system" subfolder)**

### **Why "system" is better than "core":**
1. **🔧 Clear Purpose**: "system" clearly indicates internal TARS resources
2. **👤 User-Friendly**: Users understand "system" vs "workspace" distinction
3. **🔄 Scalable**: Easy to add new system components
4. **📁 Intuitive**: Follows common software organization patterns

### **🚀 Benefits of This Organization:**

#### **🔧 System Isolation**
- **Internal TARS resources** clearly separated
- **System metascripts** protected from user modification
- **Core configuration** organized and secure
- **Engine components** logically grouped

#### **👤 User Experience**
- **Clean workspace** for user content
- **Clear boundaries** between system and user files
- **Easy navigation** with logical grouping
- **Reduced confusion** about what to modify

#### **🔄 Maintainability**
- **System updates** don't affect user workspace
- **Version control** easier with clear separation
- **Backup strategies** can target specific areas
- **Documentation** clearer with organized structure

## 📋 **MIGRATION PLAN**

### **Phase 1: Create New Structure**
1. Create `system/`, `workspace/`, `shared/` directories
2. Move files according to classification
3. Update references in configuration files
4. Test all functionality

### **Phase 2: Update References**
1. Update `tars.yaml` paths
2. Update metascript references
3. Update CLI tool paths
4. Update documentation

### **Phase 3: Validation**
1. Test all TARS functionality
2. Verify metascript execution
3. Check configuration loading
4. Validate user workflows

## 🗂️ **DETAILED FILE CLASSIFICATION**

### **🔧 SYSTEM FILES (move to system/)**
- `config/` → `system/config/`
- `metascripts/` → `system/metascripts/`
- `workflows/` → `system/workflows/`
- `scripts/` → `system/scripts/`
- `knowledge/` → `system/knowledge/`
- Core templates → `system/templates/`

### **👤 USER FILES (move to workspace/)**
- `docs/` → `workspace/docs/`
- `examples/` → `workspace/examples/`
- `explorations/` → `workspace/explorations/`
- `plans/` → `workspace/plans/`
- User templates → `workspace/templates/`

### **🤝 SHARED FILES (move to shared/)**
- Common templates → `shared/templates/`
- Reusable libraries → `shared/libraries/`
- Shared assets → `shared/assets/`

### **📁 ROOT FILES (stay at root)**
- `tars.yaml` (main configuration)
- `ORGANIZATION_SUMMARY.md` (project overview)
- `README.md` (if exists)

## 🎯 **IMPLEMENTATION PRIORITY**

### **🔥 HIGH PRIORITY**
- Move system metascripts to `system/metascripts/`
- Move TARS configuration to `system/config/`
- Move core workflows to `system/workflows/`

### **📊 MEDIUM PRIORITY**
- Move user documentation to `workspace/docs/`
- Move planning files to `workspace/plans/`
- Move examples to `workspace/examples/`

### **📝 LOW PRIORITY**
- Create shared resources structure
- Organize templates by category
- Set up user project structure

## ✅ **SUCCESS CRITERIA**

- [ ] **Clear separation** between system and user resources
- [ ] **All TARS functionality** working after migration
- [ ] **Intuitive navigation** for users and developers
- [ ] **Maintainable structure** for future development
- [ ] **Documentation updated** to reflect new organization

---

**This reorganization will create a professional, scalable, and user-friendly TARS directory structure that clearly separates internal system resources from user workspace content.**
