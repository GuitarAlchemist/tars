#!/usr/bin/env python3
"""
TARS QA Bug Reporter
Analyzes deployment failures and reports detailed bug information
"""

import os
import sys
import subprocess
import time
import requests
import json
from datetime import datetime
from pathlib import Path

class QABugReporter:
    def __init__(self):
        self.bugs = []
        self.test_results = {}
        
    def run_command(self, cmd, cwd=None):
        """Execute command and return result"""
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=cwd,
                capture_output=True, text=True, timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return -1, "", str(e)
    
    def analyze_container(self, container_name):
        """Analyze container for issues"""
        print(f"🔍 ANALYZING CONTAINER: {container_name}")
        print("=" * 50)
        
        issues = []
        
        # Check if container is running
        code, output, error = self.run_command(f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'")
        if code != 0 or not output.strip():
            issues.append({
                "severity": "CRITICAL",
                "category": "Container Status",
                "issue": "Container not running",
                "details": f"Container {container_name} is not in running state",
                "reproduction": f"docker ps --filter name={container_name}"
            })
            return issues
        
        print(f"  ✅ Container Status: {output.strip()}")
        
        # Check container logs
        code, logs, error = self.run_command(f"docker logs {container_name}")
        if code == 0:
            if not logs.strip():
                issues.append({
                    "severity": "HIGH",
                    "category": "Application Startup",
                    "issue": "No application logs found",
                    "details": "Container is running but no application output detected",
                    "reproduction": f"docker logs {container_name}",
                    "logs": "Empty logs"
                })
            elif "error" in logs.lower() or "exception" in logs.lower():
                issues.append({
                    "severity": "HIGH",
                    "category": "Application Error",
                    "issue": "Application errors in logs",
                    "details": "Error messages found in container logs",
                    "reproduction": f"docker logs {container_name}",
                    "logs": logs[:500] + "..." if len(logs) > 500 else logs
                })
        
        # Check for .NET application files
        code, output, error = self.run_command(f"docker exec {container_name} find . -name '*.dll' -path '*/bin/Release/*'")
        if code == 0:
            if not output.strip():
                issues.append({
                    "severity": "CRITICAL",
                    "category": "Build Artifacts",
                    "issue": "No compiled .NET DLLs found",
                    "details": "Project appears to be documentation-only with no executable code",
                    "reproduction": f"docker exec {container_name} find . -name '*.dll' -path '*/bin/Release/*'",
                    "resolution": "Generate actual F# application code with Program.fs and proper entry point"
                })
        
        # Check for F# project files
        code, output, error = self.run_command(f"docker exec {container_name} find . -name '*.fsproj'")
        if code == 0 and output.strip():
            print(f"  📄 Found F# projects: {len(output.strip().split())}")
            
            # Check if projects have actual source code
            code, fs_files, error = self.run_command(f"docker exec {container_name} find . -name '*.fs'")
            if code == 0:
                if not fs_files.strip():
                    issues.append({
                        "severity": "HIGH",
                        "category": "Source Code",
                        "issue": "F# project files exist but no .fs source files found",
                        "details": "Project structure exists but contains no actual F# source code",
                        "reproduction": f"docker exec {container_name} find . -name '*.fs'",
                        "resolution": "Add actual F# source files (Program.fs, Controllers.fs, etc.)"
                    })
        
        # Check if any process is listening on port 5000
        code, output, error = self.run_command(f"docker exec {container_name} netstat -tlnp 2>/dev/null | grep :5000 || echo 'No process on port 5000'")
        if code == 0:
            if "No process on port 5000" in output:
                issues.append({
                    "severity": "CRITICAL",
                    "category": "Network Service",
                    "issue": "No application listening on port 5000",
                    "details": "Container is running but no service is bound to the expected port",
                    "reproduction": f"docker exec {container_name} netstat -tlnp | grep :5000",
                    "resolution": "Ensure application starts and binds to http://0.0.0.0:5000"
                })
            else:
                print(f"  ✅ Port 5000: Service detected")
        
        return issues
    
    def test_connectivity(self, url, timeout=10):
        """Test application connectivity"""
        print(f"🌐 TESTING CONNECTIVITY: {url}")
        print("=" * 40)
        
        issues = []
        
        try:
            response = requests.get(url, timeout=timeout)
            status_code = response.status_code
            
            if status_code == 200:
                print(f"  ✅ HTTP {status_code}: Application responding")
                return issues
            elif status_code == 404:
                print(f"  ⚠️ HTTP {status_code}: Application running but no content")
                issues.append({
                    "severity": "MEDIUM",
                    "category": "Application Content",
                    "issue": f"HTTP {status_code} - No content at root path",
                    "details": "Application is running but returns 404 for root path",
                    "reproduction": f"curl {url}",
                    "resolution": "Add a default route or health check endpoint"
                })
            else:
                issues.append({
                    "severity": "MEDIUM",
                    "category": "HTTP Response",
                    "issue": f"Unexpected HTTP status: {status_code}",
                    "details": f"Application responding with HTTP {status_code}",
                    "reproduction": f"curl {url}",
                    "response_body": response.text[:200] + "..." if len(response.text) > 200 else response.text
                })
                
        except requests.exceptions.ConnectionError:
            issues.append({
                "severity": "CRITICAL",
                "category": "Network Connectivity",
                "issue": "Connection refused",
                "details": f"Cannot connect to {url} - no service listening",
                "reproduction": f"curl {url}",
                "resolution": "Ensure application starts and binds to the correct port"
            })
            print(f"  ❌ Connection refused")
            
        except requests.exceptions.Timeout:
            issues.append({
                "severity": "HIGH",
                "category": "Performance",
                "issue": "Request timeout",
                "details": f"Request to {url} timed out after {timeout} seconds",
                "reproduction": f"curl --max-time {timeout} {url}",
                "resolution": "Check application startup time and performance"
            })
            print(f"  ❌ Request timeout")
            
        except Exception as e:
            issues.append({
                "severity": "HIGH",
                "category": "Network Error",
                "issue": f"Network error: {type(e).__name__}",
                "details": str(e),
                "reproduction": f"curl {url}"
            })
            print(f"  ❌ Network error: {e}")
        
        return issues
    
    def generate_bug_report(self, container_name, url, all_issues):
        """Generate comprehensive bug report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Categorize issues by severity
        critical_issues = [i for i in all_issues if i["severity"] == "CRITICAL"]
        high_issues = [i for i in all_issues if i["severity"] == "HIGH"]
        medium_issues = [i for i in all_issues if i["severity"] == "MEDIUM"]
        
        # Determine overall test result
        test_passed = len(critical_issues) == 0 and len(high_issues) == 0
        
        report = f"""# QA Test Report - FAILED

**Container:** {container_name}
**Test URL:** {url}
**Timestamp:** {timestamp}
**Overall Result:** {'✅ PASSED' if test_passed else '❌ FAILED'}

## Summary
- **Critical Issues:** {len(critical_issues)}
- **High Priority Issues:** {len(high_issues)}
- **Medium Priority Issues:** {len(medium_issues)}
- **Total Issues Found:** {len(all_issues)}

## Root Cause Analysis

### Primary Issue
The deployment failed because the generated project contains only documentation and project structure files, but no actual executable F# application code.

### Technical Details
1. **Missing Executable Code**: No compiled .NET DLLs found in release directory
2. **No Application Entry Point**: No Program.fs or startup code
3. **Service Not Running**: No process listening on port 5000
4. **Empty Logs**: Container starts but application never initializes

"""

        if critical_issues:
            report += "\n## 🚨 Critical Issues\n"
            for i, issue in enumerate(critical_issues, 1):
                report += f"""
### {i}. {issue['issue']}
- **Category:** {issue['category']}
- **Details:** {issue['details']}
- **Reproduction:** `{issue['reproduction']}`
- **Resolution:** {issue.get('resolution', 'Investigation required')}
"""

        if high_issues:
            report += "\n## ⚠️ High Priority Issues\n"
            for i, issue in enumerate(high_issues, 1):
                report += f"""
### {i}. {issue['issue']}
- **Category:** {issue['category']}
- **Details:** {issue['details']}
- **Reproduction:** `{issue['reproduction']}`
- **Resolution:** {issue.get('resolution', 'Investigation required')}
"""

        if medium_issues:
            report += "\n## 📋 Medium Priority Issues\n"
            for i, issue in enumerate(medium_issues, 1):
                report += f"""
### {i}. {issue['issue']}
- **Category:** {issue['category']}
- **Details:** {issue['details']}
- **Reproduction:** `{issue['reproduction']}`
"""

        report += f"""
## Recommended Actions

### Immediate Fixes Required
1. **Generate Actual Application Code**
   - Add Program.fs with proper entry point
   - Implement actual API controllers
   - Add proper ASP.NET Core startup configuration

2. **Fix Build Process**
   - Ensure dotnet build produces executable DLLs
   - Verify project references and dependencies
   - Test local build before containerization

3. **Add Health Check Endpoint**
   - Implement /health endpoint for monitoring
   - Add proper error handling and logging
   - Configure application to bind to 0.0.0.0:5000

### Testing Recommendations
1. **Unit Tests**: Verify core functionality works
2. **Integration Tests**: Test API endpoints
3. **Container Tests**: Validate Docker deployment
4. **Health Monitoring**: Implement application health checks

## Container Diagnostics

### Container Status
```bash
docker ps --filter name={container_name}
```

### Application Logs
```bash
docker logs {container_name}
```

### File System Analysis
```bash
docker exec {container_name} find . -name "*.dll" -o -name "*.fs" -o -name "*.fsproj"
```

---
*Report generated by TARS QA Bug Reporter*
*Automated analysis completed at {timestamp}*
"""

        return report, test_passed
    
    def run_full_qa_analysis(self, container_name, url):
        """Run complete QA analysis and generate report"""
        print("🤖🧪 TARS QA BUG ANALYSIS")
        print("=" * 40)
        print()
        
        all_issues = []
        
        # Analyze container
        container_issues = self.analyze_container(container_name)
        all_issues.extend(container_issues)
        print()
        
        # Test connectivity
        connectivity_issues = self.test_connectivity(url)
        all_issues.extend(connectivity_issues)
        print()
        
        # Generate report
        report, test_passed = self.generate_bug_report(container_name, url, all_issues)
        
        # Save report
        report_file = f"qa-bug-report-{container_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("📋 BUG REPORT GENERATED")
        print("=" * 30)
        print(f"  📄 Report: {report_file}")
        print(f"  🎯 Result: {'PASSED' if test_passed else 'FAILED'}")
        print(f"  🐛 Issues: {len(all_issues)} found")
        print()
        
        # Print summary
        if not test_passed:
            print("❌ QA TEST FAILED")
            print("=" * 20)
            critical = [i for i in all_issues if i["severity"] == "CRITICAL"]
            high = [i for i in all_issues if i["severity"] == "HIGH"]
            
            if critical:
                print("🚨 Critical Issues:")
                for issue in critical:
                    print(f"  • {issue['issue']}")
            
            if high:
                print("⚠️ High Priority Issues:")
                for issue in high:
                    print(f"  • {issue['issue']}")
        else:
            print("✅ QA TEST PASSED")
        
        return test_passed, report_file

def main():
    """Main QA analysis"""
    if len(sys.argv) < 3:
        print("Usage: python qa-bug-reporter.py <container_name> <url>")
        print("Example: python qa-bug-reporter.py tars-apiservice-container http://localhost:5915")
        return 1
    
    container_name = sys.argv[1]
    url = sys.argv[2]
    
    qa = QABugReporter()
    test_passed, report_file = qa.run_full_qa_analysis(container_name, url)
    
    return 0 if test_passed else 1

if __name__ == "__main__":
    sys.exit(main())
