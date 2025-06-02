
def generate_test_suite(module_info):
    """Generate comprehensive test suite automatically"""
    
    test_code = f"""
module {module_info['name']}Tests

open Xunit
open FsUnit.Xunit

// Generated unit tests
{generate_unit_tests(module_info['functions'])}

// Generated property tests
{generate_property_tests(module_info['types'])}

// Generated integration tests
{generate_integration_tests(module_info['dependencies'])}

// Generated performance tests
{generate_performance_tests(module_info['critical_paths'])}
"""
    
    return compile_test_suite(test_code)

# Example: Generate tests for data source module
test_suite = generate_test_suite({
    "name": "DataSource",
    "functions": ["detect", "generate", "compile"],
    "types": ["DataSourceType", "DetectionResult"],
    "dependencies": ["HttpClient", "Database"],
    "critical_paths": ["pattern_matching", "code_compilation"]
})
