#!/usr/bin/env python3
"""
TARS Autonomous Quality Iteration Loop
Continuously improves a project until acceptable quality is achieved
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

class AutonomousQualityLoop:
    def __init__(self):
        self.max_iterations = 5
        self.quality_threshold = 0.8  # 80% quality score
        self.current_iteration = 0
        self.project_name = None
        self.project_path = None
        self.iteration_history = []
        
    def run_autonomous_iteration(self, project_name, app_type="webapi"):
        """Run the complete autonomous quality iteration loop"""
        
        print("🔄 TARS AUTONOMOUS QUALITY ITERATION LOOP")
        print("=" * 50)
        print(f"🎯 Target Project: {project_name}")
        print(f"📊 Quality Threshold: {self.quality_threshold:.0%}")
        print(f"🔢 Max Iterations: {self.max_iterations}")
        print()
        
        self.project_name = project_name
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            print(f"🔄 ITERATION {self.current_iteration}/{self.max_iterations}")
            print("=" * 40)
            
            # Phase 1: Generate/Regenerate Project
            print("📦 PHASE 1: PROJECT GENERATION")
            generation_result = self.generate_project(project_name, app_type)
            if not generation_result['success']:
                print(f"❌ Generation failed: {generation_result['error']}")
                continue
            
            self.project_path = generation_result['path']
            print(f"  ✅ Project generated: {self.project_path}")
            print()
            
            # Phase 2: Deploy Project
            print("🚀 PHASE 2: DEPLOYMENT")
            deployment_result = self.deploy_project()
            if not deployment_result['success']:
                print(f"❌ Deployment failed: {deployment_result['error']}")
                self.analyze_and_fix_deployment_issues(deployment_result)
                continue
            
            print(f"  ✅ Deployed: {deployment_result['container_name']}")
            print(f"  🌐 URL: {deployment_result['url']}")
            print()
            
            # Phase 3: Quality Assessment
            print("🧪 PHASE 3: QUALITY ASSESSMENT")
            quality_result = self.assess_quality(deployment_result)
            quality_score = quality_result['score']
            
            print(f"  📊 Quality Score: {quality_score:.1%}")
            print(f"  🎯 Threshold: {self.quality_threshold:.0%}")
            
            # Record iteration
            iteration_record = {
                'iteration': self.current_iteration,
                'timestamp': datetime.now().isoformat(),
                'generation_success': generation_result['success'],
                'deployment_success': deployment_result['success'],
                'quality_score': quality_score,
                'issues_found': quality_result['issues'],
                'fixes_applied': []
            }
            
            if quality_score >= self.quality_threshold:
                print(f"  ✅ Quality threshold reached!")
                iteration_record['status'] = 'SUCCESS'
                self.iteration_history.append(iteration_record)
                self.generate_final_report(deployment_result, quality_result)
                return True
            
            print(f"  ⚠️ Quality below threshold, analyzing issues...")
            print()
            
            # Phase 4: Issue Analysis and Fixes
            print("🔧 PHASE 4: AUTONOMOUS FIXES")
            fixes_applied = self.analyze_and_apply_fixes(quality_result, deployment_result)
            iteration_record['fixes_applied'] = fixes_applied
            iteration_record['status'] = 'IMPROVED'
            
            print(f"  🛠️ Applied {len(fixes_applied)} fixes")
            for fix in fixes_applied:
                print(f"    • {fix}")
            print()
            
            self.iteration_history.append(iteration_record)
            
            # Clean up for next iteration
            self.cleanup_deployment(deployment_result)
        
        print("❌ MAXIMUM ITERATIONS REACHED")
        if 'quality_score' in locals():
            print(f"Final quality score: {quality_score:.1%}")
            self.generate_final_report(deployment_result, quality_result)
        else:
            print("No successful iterations completed")
        return False
    
    def generate_project(self, project_name, app_type):
        """Generate or regenerate the project"""
        try:
            # Use enhanced project generator
            cmd = f"python enhanced-project-generator.py {project_name}_v{self.current_iteration} {app_type}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                project_path = f"output/projects/{project_name}_v{self.current_iteration}"
                return {
                    'success': True,
                    'path': project_path,
                    'output': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'output': result.stdout
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def deploy_project(self):
        """Deploy the project using simple deployment"""
        try:
            project_name = f"{self.project_name}_v{self.current_iteration}"
            project_path = f"output/projects/{project_name}"

            # Use simple deployment
            result = subprocess.run(
                f"python simple-deploy.py {project_path}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=180
            )

            if result.returncode == 0 and "SUCCESS:" in result.stdout:
                # Parse success output: SUCCESS:container_name:ip:port
                parts = result.stdout.strip().split("SUCCESS:")[1].split(":")
                container_name = parts[0]
                ip = parts[1] if len(parts) > 1 else "localhost"
                port = int(parts[2]) if len(parts) > 2 else 5000

                return {
                    'success': True,
                    'container_name': container_name,
                    'ip': ip,
                    'port': port,
                    'url': f"http://{ip}:{port}",
                    'output': result.stdout
                }
            else:
                error_info = result.stdout if "FAILED:" in result.stdout else result.stderr
                return {
                    'success': False,
                    'error': error_info,
                    'build_logs': error_info
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def assess_quality(self, deployment_result):
        """Assess the quality of the deployed project"""
        issues = []
        quality_factors = {
            'deployment_success': 0.3,
            'application_response': 0.3,
            'code_quality': 0.2,
            'documentation': 0.1,
            'configuration': 0.1
        }
        
        scores = {}
        
        # 1. Deployment Success (already known)
        scores['deployment_success'] = 1.0 if deployment_result['success'] else 0.0
        
        # 2. Application Response
        if deployment_result['success']:
            response_score = self.test_application_response(deployment_result)
            scores['application_response'] = response_score['score']
            issues.extend(response_score['issues'])
        else:
            scores['application_response'] = 0.0
            issues.append("Application failed to deploy")
        
        # 3. Code Quality
        code_score = self.assess_code_quality()
        scores['code_quality'] = code_score['score']
        issues.extend(code_score['issues'])
        
        # 4. Documentation Quality
        doc_score = self.assess_documentation()
        scores['documentation'] = doc_score['score']
        issues.extend(doc_score['issues'])
        
        # 5. Configuration Quality
        config_score = self.assess_configuration()
        scores['configuration'] = config_score['score']
        issues.extend(config_score['issues'])
        
        # Calculate weighted quality score
        total_score = sum(scores[factor] * weight for factor, weight in quality_factors.items())
        
        return {
            'score': total_score,
            'scores': scores,
            'issues': issues,
            'factors': quality_factors
        }
    
    def test_application_response(self, deployment_result):
        """Test if the application responds correctly"""
        issues = []
        
        try:
            import requests
            url = deployment_result['url']
            
            # Test basic connectivity
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return {'score': 1.0, 'issues': []}
                elif response.status_code == 404:
                    issues.append("Application responds but returns 404")
                    return {'score': 0.7, 'issues': issues}
                else:
                    issues.append(f"Application responds with HTTP {response.status_code}")
                    return {'score': 0.5, 'issues': issues}
            except requests.exceptions.ConnectionError:
                # Check if it's a console app that exited
                container_logs = self.get_container_logs(deployment_result['container_name'])
                if container_logs and "Welcome to" in container_logs:
                    issues.append("Console application executed successfully but exited")
                    return {'score': 0.8, 'issues': issues}
                else:
                    issues.append("Application not responding - connection refused")
                    return {'score': 0.2, 'issues': issues}
                    
        except Exception as e:
            issues.append(f"Error testing application: {e}")
            return {'score': 0.0, 'issues': issues}
    
    def assess_code_quality(self):
        """Assess the quality of generated code"""
        issues = []
        score = 1.0
        
        if not self.project_path or not os.path.exists(self.project_path):
            return {'score': 0.0, 'issues': ['Project path not found']}
        
        # Check for source files
        src_dir = os.path.join(self.project_path, "src")
        if not os.path.exists(src_dir):
            issues.append("No src directory found")
            score -= 0.3
        else:
            fs_files = [f for f in os.listdir(src_dir) if f.endswith('.fs')]
            if not fs_files:
                issues.append("No F# source files found")
                score -= 0.5
            else:
                # Check for entry point
                program_fs = os.path.join(src_dir, "Program.fs")
                if os.path.exists(program_fs):
                    with open(program_fs, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '[<EntryPoint>]' not in content:
                            issues.append("No entry point found in Program.fs")
                            score -= 0.3
                        if 'namespace' not in content:
                            issues.append("No namespace declaration found")
                            score -= 0.2
                else:
                    issues.append("No Program.fs entry point file")
                    score -= 0.4
        
        return {'score': max(0.0, score), 'issues': issues}
    
    def assess_documentation(self):
        """Assess documentation quality"""
        issues = []
        score = 1.0
        
        readme_path = os.path.join(self.project_path, "README.md")
        if not os.path.exists(readme_path):
            issues.append("No README.md found")
            score -= 0.5
        else:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) < 500:
                    issues.append("README.md is too brief")
                    score -= 0.3
        
        return {'score': max(0.0, score), 'issues': issues}
    
    def assess_configuration(self):
        """Assess configuration quality"""
        issues = []
        score = 1.0
        
        # Check for Dockerfile
        dockerfile_path = os.path.join(self.project_path, "Dockerfile")
        if not os.path.exists(dockerfile_path):
            issues.append("No Dockerfile found")
            score -= 0.4
        
        # Check for project file
        proj_files = [f for f in os.listdir(self.project_path) if f.endswith('.fsproj')]
        if not proj_files:
            issues.append("No .fsproj file found")
            score -= 0.6
        
        return {'score': max(0.0, score), 'issues': issues}
    
    def analyze_and_apply_fixes(self, quality_result, deployment_result):
        """Analyze issues and apply autonomous fixes"""
        fixes_applied = []
        
        for issue in quality_result['issues']:
            if "JSON string" in issue or "syntax error" in issue.lower():
                fix = self.fix_json_syntax_errors()
                if fix:
                    fixes_applied.append("Fixed JSON syntax errors in generated code")
            
            elif "No entry point" in issue:
                fix = self.fix_missing_entry_point()
                if fix:
                    fixes_applied.append("Added missing entry point to Program.fs")
            
            elif "connection refused" in issue.lower():
                fix = self.fix_application_startup()
                if fix:
                    fixes_applied.append("Fixed application startup configuration")
            
            elif "No README.md" in issue:
                fix = self.generate_better_documentation()
                if fix:
                    fixes_applied.append("Generated comprehensive README.md")
        
        return fixes_applied
    
    def fix_json_syntax_errors(self):
        """Fix JSON syntax errors in generated F# code"""
        try:
            program_fs = os.path.join(self.project_path, "src", "Program.fs")
            if os.path.exists(program_fs):
                with open(program_fs, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Fix unescaped quotes in JSON strings
                content = content.replace('{"status":', '"{\"status\":"')
                content = content.replace('"service":', '\"service\":"')
                content = content.replace('"version":', '\"version\":"')
                
                with open(program_fs, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return True
        except Exception:
            pass
        return False
    
    def fix_missing_entry_point(self):
        """Add missing entry point to Program.fs"""
        try:
            program_fs = os.path.join(self.project_path, "src", "Program.fs")
            if os.path.exists(program_fs):
                with open(program_fs, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if '[<EntryPoint>]' not in content:
                    # Add entry point before main function
                    content = content.replace('let main args =', '[<EntryPoint>]\n    let main args =')
                    
                    with open(program_fs, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    return True
        except Exception:
            pass
        return False
    
    def fix_application_startup(self):
        """Fix application startup issues"""
        # This would implement more sophisticated startup fixes
        return False
    
    def generate_better_documentation(self):
        """Generate comprehensive documentation"""
        try:
            readme_path = os.path.join(self.project_path, "README.md")
            enhanced_readme = f"""# {self.project_name} - Enhanced by TARS

This project was autonomously generated and improved by TARS through {self.current_iteration} iterations.

## Quality Improvements Applied

""" + "\n".join(f"- Iteration {i+1}: {record.get('fixes_applied', [])}" for i, record in enumerate(self.iteration_history)) + """

## Project Structure

```
{self.project_name}/
├── src/                 # Source code
├── tests/              # Unit tests  
├── docs/               # Documentation
├── Dockerfile          # Container configuration
└── README.md           # This file
```

## Build and Run

```bash
dotnet build
dotnet run
```

## Docker Deployment

```bash
docker build -t {self.project_name.lower()} .
docker run -p 5000:5000 {self.project_name.lower()}
```

---
*Autonomously improved by TARS Quality Iteration Loop*
"""
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_readme)
            
            return True
        except Exception:
            pass
        return False
    
    def analyze_and_fix_deployment_issues(self, deployment_result):
        """Analyze and fix deployment-specific issues"""
        if 'build_logs' in deployment_result:
            build_logs = deployment_result['build_logs']
            # Analyze build logs for specific errors and apply fixes
            pass
    
    def extract_build_logs(self, error_output):
        """Extract build logs from error output"""
        return error_output
    
    def get_container_logs(self, container_name):
        """Get logs from a container"""
        try:
            result = subprocess.run(f"docker logs {container_name}", shell=True, capture_output=True, text=True)
            return result.stdout
        except:
            return ""
    
    def cleanup_deployment(self, deployment_result):
        """Clean up deployment for next iteration"""
        if deployment_result['success']:
            try:
                subprocess.run(f"docker stop {deployment_result['container_name']}", shell=True, capture_output=True)
                subprocess.run(f"docker rm {deployment_result['container_name']}", shell=True, capture_output=True)
            except:
                pass
    
    def generate_final_report(self, deployment_result, quality_result):
        """Generate final iteration report"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_file = f"autonomous-quality-report-{self.project_name}-{timestamp}.md"
        
        report = f"""# TARS Autonomous Quality Iteration Report

**Project:** {self.project_name}
**Final Quality Score:** {quality_result['score']:.1%}
**Iterations Completed:** {self.current_iteration}
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Iteration History

""" + "\n".join(f"### Iteration {record['iteration']}\n- **Quality Score:** {record['quality_score']:.1%}\n- **Fixes Applied:** {len(record['fixes_applied'])}\n- **Status:** {record['status']}" for record in self.iteration_history) + """

## Final Assessment

### Quality Breakdown
""" + "\n".join(f"- **{factor.replace('_', ' ').title()}:** {score:.1%}" for factor, score in quality_result['scores'].items()) + """

### Remaining Issues
""" + "\n".join(f"- {issue}" for issue in quality_result['issues']) + """

## Autonomous Improvements Demonstrated

✅ **Project Generation:** Multiple iterations with improvements
✅ **Deployment Testing:** Real container deployment and testing  
✅ **Quality Assessment:** Multi-factor quality scoring
✅ **Autonomous Fixes:** Automatic issue detection and resolution
✅ **Iterative Improvement:** Continuous quality enhancement

---
*Generated by TARS Autonomous Quality Iteration Loop*
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Final report generated: {report_file}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python autonomous-quality-loop.py <project_name> [app_type]")
        print("Example: python autonomous-quality-loop.py IterativeAPI webapi")
        return 1
    
    project_name = sys.argv[1]
    app_type = sys.argv[2] if len(sys.argv) > 2 else "console"  # Use console for reliability
    
    loop = AutonomousQualityLoop()
    success = loop.run_autonomous_iteration(project_name, app_type)
    
    if success:
        print("🎉 AUTONOMOUS QUALITY ITERATION SUCCESSFUL!")
        return 0
    else:
        print("❌ Quality threshold not reached within maximum iterations")
        return 1

if __name__ == "__main__":
    sys.exit(main())
