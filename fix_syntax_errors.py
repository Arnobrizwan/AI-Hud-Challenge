#!/usr/bin/env python3
"""
Script to fix syntax errors in Python files
"""
import os
import re
import subprocess
import sys


def fix_syntax_errors(file_path: str) -> bool:
    """Fix syntax errors in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix the main pattern: ) -> Dict[str, Any]:
        # """docstring"""
        patterns = [
            # Pattern 1: ) -> Dict[str, Any]:
            # """docstring"""
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*"""([^"]*)"""', r') -> Dict[str, Any]:\n    """\1"""'),
            
            # Pattern 2: ) -> Dict[str, Any]:
    if scope["type"] == "http":
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*if\s+scope\["type"\]\s*==\s*"http":', r') -> Dict[str, Any]:\n    if scope["type"] == "http":'),
            
            # Pattern 3: ) -> Dict[str, Any]:
    pass
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*pass', r') -> Dict[str, Any]:\n    pass'),
            
            # Pattern 4: ) -> Dict[str, Any]:
    start_time = time.time()
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*start_time\s*=\s*time\.time\(\)', r') -> Dict[str, Any]:\n    start_time = time.time()'),
            
            # Pattern 5: ) -> Dict[str, Any]:
    """docstring"""
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*"""([^"]*)"""\s*\n', r') -> Dict[str, Any]:\n    """\1"""\n'),
            
            # Pattern 6: ) -> Dict[str, Any]:
    """docstring"""
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*"""([^"]*)"""\s*$', r') -> Dict[str, Any]:\n    """\1"""'),
            
            # Pattern 7: ) -> Dict[str, Any]:
    """docstring""" """
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*"""([^"]*)"""\s*\n\s*"""', r') -> Dict[str, Any]:\n    """\1"""\n    """'),
            
            # Pattern 8: ) -> Dict[str, Any]:
    """docstring""" """docstring2"""
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*"""([^"]*)"""\s*\n\s*"""([^"]*)"""', r') -> Dict[str, Any]:\n    """\1"""\n    """\2"""'),
            
            # Pattern 9: ) -> Dict[str, Any]:
    """docstring""" """docstring2"""
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*"""([^"]*)"""\s*\n\s*"""([^"]*)"""\s*\n', r') -> Dict[str, Any]:\n    """\1"""\n    """\2"""\n'),
            
            # Pattern 10: ) -> Dict[str, Any]:
    """docstring""" """docstring2"""
            (r'\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*"""([^"]*)"""\s*\n\s*"""([^"]*)"""\s*$', r') -> Dict[str, Any]:\n    """\1"""\n    """\2"""'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return False

def find_python_files(directory: str):
    """Find all Python files in directory"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    """Main function to fix all errors"""
    base_dir = "/Users/arnobrizwan/AI-Hud-Challenge"
    
    # Find all Python files
    python_files = find_python_files(base_dir)
    
    print(f"Found {len(python_files)} Python files")
    
    # Fix syntax errors
    print("\n=== Fixing syntax errors ===")
    syntax_fixed = 0
    for file_path in python_files:
        if fix_syntax_errors(file_path):
            print(f"Fixed syntax: {file_path}")
            syntax_fixed += 1
    
    print(f"Fixed syntax in {syntax_fixed} files")

if __name__ == "__main__":
    main()
