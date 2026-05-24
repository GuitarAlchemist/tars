#!/usr/bin/env python3
"""
Integration test for TARS Project Pipeline feature (Phase 10).

Tests:
1. Create a new project with SDLC template
2. List projects
3. Check project status  
4. Run pipeline
5. Generate demo output

Usage:
    python tests/integration/test_project_pipeline.py
"""

import subprocess
import sys
import os
import json
from pathlib import Path

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def run_tars(*args):
    """Run TARS CLI command and return (exit_code, stdout, stderr)."""
    cmd = ["dotnet", "run", "--project", "src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj", "--"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def test_pipeline_new():
    """Test creating a new project."""
    print(f"\n{YELLOW}Test 1: Create new project{RESET}")
    
    code, out, err = run_tars("pipeline", "new", "test-proj-001", "-n", "Test Project", "-t", "sdlc", "-m", "continuous")
    
    if code == 0 and "Created project" in out:
        print(f"  {GREEN}✓ Project created successfully{RESET}")
        print(f"    Output: {out.strip()[:100]}...")
        return True
    else:
        print(f"  {RED}✗ Failed to create project{RESET}")
        print(f"    Exit code: {code}")
        print(f"    Stdout: {out}")
        print(f"    Stderr: {err}")
        return False

def test_pipeline_list():
    """Test listing projects."""
    print(f"\n{YELLOW}Test 2: List projects{RESET}")
    
    code, out, err = run_tars("pipeline", "list")
    
    if code == 0 and ("test-proj-001" in out or "Projects" in out):
        print(f"  {GREEN}✓ Listed projects successfully{RESET}")
        return True
    else:
        print(f"  {RED}✗ Failed to list projects{RESET}")
        print(f"    Exit code: {code}")
        print(f"    Output: {out}")
        return False

def test_pipeline_status():
    """Test project status."""
    print(f"\n{YELLOW}Test 3: Get project status{RESET}")
    
    code, out, err = run_tars("pipeline", "status", "test-proj-001")
    
    if code == 0 and ("Project:" in out or "Template:" in out):
        print(f"  {GREEN}✓ Got project status{RESET}")
        return True
    else:
        print(f"  {RED}✗ Failed to get status{RESET}")
        print(f"    Exit code: {code}")
        print(f"    Output: {out}")
        return False

def test_pipeline_run():
    """Test running pipeline."""
    print(f"\n{YELLOW}Test 4: Run pipeline{RESET}")
    
    code, out, err = run_tars("pipeline", "run", "test-proj-001")
    
    if code == 0 and ("Pipeline" in out or "Stage" in out or "completed" in out.lower()):
        print(f"  {GREEN}✓ Pipeline executed{RESET}")
        return True
    else:
        print(f"  {RED}✗ Failed to run pipeline{RESET}")
        print(f"    Exit code: {code}")
        print(f"    Output: {out}")
        return False

def test_pipeline_demo_markdown():
    """Test demo generation in Markdown format."""
    print(f"\n{YELLOW}Test 5: Generate Markdown demo{RESET}")
    
    code, out, err = run_tars("pipeline", "demo", "test-proj-001", "-f", "markdown")
    
    if code == 0 and ("#" in out or "Test Project" in out):
        print(f"  {GREEN}✓ Generated Markdown demo{RESET}")
        print(f"    Preview: {out[:200]}...")
        return True
    else:
        print(f"  {RED}✗ Failed to generate demo{RESET}")
        print(f"    Exit code: {code}")
        print(f"    Output: {out}")
        return False

def test_pipeline_demo_json():
    """Test demo generation in JSON format."""
    print(f"\n{YELLOW}Test 6: Generate JSON demo{RESET}")
    
    code, out, err = run_tars("pipeline", "demo", "test-proj-001", "-f", "json")
    
    if code == 0:
        try:
            # Try to parse as JSON
            data = json.loads(out)
            if "projectId" in data and "sections" in data:
                print(f"  {GREEN}✓ Generated valid JSON demo{RESET}")
                return True
        except json.JSONDecodeError:
            pass
    
    # Fallback: check for JSON-like structure
    if code == 0 and "{" in out and "projectId" in out:
        print(f"  {GREEN}✓ Generated JSON demo{RESET}")
        return True
    
    print(f"  {RED}✗ Failed to generate JSON demo{RESET}")
    print(f"    Exit code: {code}")
    print(f"    Output: {out[:200]}")
    return False

def test_pipeline_demo_html():
    """Test demo generation in HTML format."""
    print(f"\n{YELLOW}Test 7: Generate HTML demo{RESET}")
    
    code, out, err = run_tars("pipeline", "demo", "test-proj-001", "-f", "html")
    
    if code == 0 and ("<!DOCTYPE html>" in out or "<html>" in out):
        print(f"  {GREEN}✓ Generated HTML demo{RESET}")
        return True
    else:
        print(f"  {RED}✗ Failed to generate HTML demo{RESET}")
        print(f"    Exit code: {code}")
        print(f"    Output: {out[:200]}")
        return False

def test_agile_template():
    """Test creating project with Agile template."""
    print(f"\n{YELLOW}Test 8: Create Agile project{RESET}")
    
    code, out, err = run_tars("pipeline", "new", "agile-proj", "-n", "Agile Project", "-t", "agile", "-m", "hitl")
    
    if code == 0 and "Created project" in out:
        print(f"  {GREEN}✓ Agile project created{RESET}")
        return True
    else:
        print(f"  {RED}✗ Failed to create Agile project{RESET}")
        return False

def test_research_template():
    """Test creating project with Research template."""
    print(f"\n{YELLOW}Test 9: Create Research project{RESET}")
    
    code, out, err = run_tars("pipeline", "new", "research-proj", "-n", "Research Project", "-t", "research", "-m", "hybrid")
    
    if code == 0 and "Created project" in out:
        print(f"  {GREEN}✓ Research project created{RESET}")
        return True
    else:
        print(f"  {RED}✗ Failed to create Research project{RESET}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("TARS Project Pipeline Integration Tests (Phase 10)")
    print("=" * 60)
    
    # Change to repo root
    os.chdir(Path(__file__).parent.parent.parent)
    
    tests = [
        test_pipeline_new,
        test_pipeline_list,
        test_pipeline_status,
        test_pipeline_run,
        test_pipeline_demo_markdown,
        test_pipeline_demo_json,
        test_pipeline_demo_html,
        test_agile_template,
        test_research_template,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {RED}✗ Exception: {e}{RESET}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {GREEN}{passed} passed{RESET}, {RED if failed else ''}{failed} failed{RESET}")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
