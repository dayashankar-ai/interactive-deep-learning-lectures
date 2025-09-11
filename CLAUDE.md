# Claude Code Assistant Documentation
## Deep Learning Course Management Guide

This document outlines essential commands and workflows for managing the Interactive Deep Learning Lectures repository using Claude Code Assistant.

---

## 🚀 Quick Start Commands

### **Git Operations**

#### Check Repository Status
```bash
git status
```
- Shows modified, staged, and untracked files
- Essential before any commit operation

#### Stage Files for Commit
```bash
# Stage specific file
git add filename.html

# Stage all changes
git add -A

# Stage multiple files
git add file1.html file2.html
```

#### Commit Changes
```bash
git commit -m "Your commit message here"

# Multi-line commit with detailed description
git commit -m "$(cat <<'EOF'
Title: Brief description of changes

- Detailed change 1
- Detailed change 2
- Impact and reasoning

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

#### Push to GitHub
```bash
git push origin main
```

#### Complete Workflow (Stage → Commit → Push)
```bash
git add filename.html && git commit -m "Update filename" && git push origin main
```

---

## 📁 Repository Structure

### **Directory Organization**
```
interactive-deep-learning-lectures/
├── index.html                 # Main website homepage
├── lectures/                  # All lecture files
│   ├── lecture01-*.html
│   ├── lecture02-*.html
│   └── ...
├── labs/                      # All lab files  
│   ├── lab00-*.html
│   ├── lab01-*.html
│   └── ...
├── assets/                    # CSS, images, resources
└── CLAUDE.md                 # This documentation
```

### **File Naming Convention**
- **Lectures**: `lecture##-descriptive-name.html`
- **Labs**: `lab##-descriptive-name.html`
- Use lowercase, hyphens for spaces, sequential numbering

---

## 🔧 Essential Workflows

### **1. Adding New Lecture**

1. **Create the lecture file**:
   ```bash
   # Place in lectures/ directory
   lectures/lecture##-topic-name.html
   ```

2. **Update index.html** to add the lecture link:
   - Find the lectures section
   - Add new lecture card with proper href path
   - Update progress counters if needed

3. **Stage, commit, and push**:
   ```bash
   git add lectures/lecture##-topic-name.html index.html
   git commit -m "Add Lecture ##: Topic Name"
   git push origin main
   ```

### **2. Adding New Lab**

1. **Create the lab file**:
   ```bash
   # Place in labs/ directory  
   labs/lab##-topic-name.html
   ```

2. **Update index.html** to add the lab link:
   - Find the labs section
   - Add new lab card with proper href path
   - Ensure correct numbering and descriptions

3. **Stage, commit, and push**:
   ```bash
   git add labs/lab##-topic-name.html index.html
   git commit -m "Add Lab ##: Topic Name"
   git push origin main
   ```

### **3. Fixing Broken Links**

1. **Identify the issue**:
   ```bash
   # Check if files exist
   ls lectures/
   ls labs/
   
   # Search for links in index.html
   grep -n "href.*labs\|href.*lectures" index.html
   ```

2. **Fix the links**:
   - Edit index.html
   - Update href paths to match actual file locations
   - Verify descriptions match file content

3. **Test and deploy**:
   ```bash
   git add index.html
   git commit -m "Fix broken links in navigation"
   git push origin main
   ```

### **4. Updating Existing Content**

1. **Modify the content file**:
   - Edit lecture or lab file directly
   - Update titles, descriptions, or content

2. **Update references if needed**:
   - Check if index.html descriptions need updating
   - Update any cross-references

3. **Deploy changes**:
   ```bash
   git add modified-file.html [index.html if updated]
   git commit -m "Update Content: Description of changes"
   git push origin main
   ```

---

## 🎯 Best Practices

### **Commit Messages**
- Use clear, descriptive titles
- Include impact and reasoning for major changes
- Follow the established format with Claude Code attribution

### **File Management**
- Always check file locations before creating links
- Use consistent naming conventions
- Organize files in appropriate directories

### **Testing Before Deployment**
- Verify all links work correctly
- Check file paths match directory structure
- Ensure descriptions match actual content

### **Git Workflow**
- Always check `git status` before committing
- Stage related changes together
- Push regularly to keep repository synchronized

---

## 🔍 Debugging Common Issues

### **Problem: Lecture/Lab Not Visible on Website**
**Solution**:
1. Check if file exists in correct directory
2. Verify index.html has correct href path
3. Ensure file naming matches link

### **Problem: Wrong Content Descriptions**
**Solution**:
1. Read the actual file content
2. Update index.html descriptions to match
3. Fix any mismatched titles or objectives

### **Problem: Git Push Rejected**
**Solution**:
```bash
git pull origin main    # Get latest changes
git push origin main    # Try push again
```

### **Problem: Files in Wrong Directory**
**Solution**:
```bash
# Move file to correct location
git mv old/path/file.html new/path/file.html
git commit -m "Reorganize file structure"
git push origin main
```

---

## 📝 Quick Reference Commands

### **Repository Management**
```bash
# Clone repository
git clone https://github.com/dayashankar-ai/interactive-deep-learning-lectures.git

# Check current branch
git branch

# Pull latest changes
git pull origin main

# View commit history
git log --oneline
```

### **File Operations**
```bash
# List files
ls -la

# List specific directory
ls lectures/
ls labs/

# Search for text in files
grep -r "search term" .

# Check file differences
git diff filename.html
```

### **Verification Commands**
```bash
# Verify all lab files exist
ls -1 labs/ | wc -l

# Check all links in index.html
grep -o 'href="[^"]*"' index.html | sort

# Verify file structure
find . -name "*.html" | head -20
```

---

## 🌐 Website Publication

The website is automatically published through GitHub Pages when changes are pushed to the main branch. Changes become live within minutes of pushing to GitHub.

**Publication Flow**:
1. Local changes → `git add` → `git commit` → `git push`
2. GitHub receives changes
3. Website updates automatically
4. Changes visible at repository URL

---

## 💡 Pro Tips

1. **Batch Related Changes**: Group related file updates in single commits
2. **Descriptive Commit Messages**: Help track changes and debug issues
3. **Regular Pushes**: Keep repository synchronized and backed up
4. **Verify Before Push**: Always check git status and file paths
5. **Consistent Structure**: Maintain naming conventions and directory organization

---

## 🆘 Emergency Procedures

### **If Something Goes Wrong**
1. **Don't Panic**: Git tracks all changes
2. **Check Status**: `git status` shows current state
3. **View Differences**: `git diff` shows what changed
4. **Revert if Needed**: `git restore filename` to undo local changes
5. **Get Help**: Use git log to see recent commits

### **Recovery Commands**
```bash
# Undo local changes (not committed)
git restore filename.html

# Reset to last commit
git reset --hard HEAD

# View what will be committed
git diff --cached
```

---

*This documentation is maintained by Claude Code Assistant for the Interactive Deep Learning Lectures project.*

**Last Updated**: September 2025  
**Repository**: https://github.com/dayashankar-ai/interactive-deep-learning-lectures