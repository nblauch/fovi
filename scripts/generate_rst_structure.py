#!/usr/bin/env python3
"""
Script to automatically generate RST files for all packages and modules.
This ensures the documentation structure is always up-to-date with the codebase.
"""

import os
import pkgutil
import importlib
import subprocess
from pathlib import Path

def find_all_modules(package_name, package_path):
    """Recursively find all modules and subpackages."""
    modules = []
    subpackages = []
    
    for finder, name, ispkg in pkgutil.iter_modules([package_path]):
        full_name = f"{package_name}.{name}"
        if ispkg:
            subpackages.append((full_name, name))
            # Recursively find modules in subpackages
            subpackage_path = os.path.join(package_path, name)
            sub_modules, sub_subpackages = find_all_modules(full_name, subpackage_path)
            modules.extend(sub_modules)
            subpackages.extend(sub_subpackages)
        else:
            modules.append((full_name, name))
    
    return modules, subpackages

def generate_package_rst(package_name, package_path, output_dir):
    """Generate RST file for a package."""
    modules, subpackages = find_all_modules(package_name, package_path)
    
    # Create the RST content
    rst_content = f"""{package_name} package
{'=' * len(package_name + ' package')}

.. automodule:: {package_name}
   :members:
   :undoc-members:
   :show-inheritance:

"""
    
    # Add subpackages if any
    if subpackages:
        rst_content += ".. toctree::\n   :maxdepth: 4\n   :caption: Subpackages\n\n"
        for full_name, name in subpackages:
            rst_content += f"   {full_name}\n"
        rst_content += "\n"
    
    # Add modules if any
    if modules:
        rst_content += ".. toctree::\n   :maxdepth: 4\n   :caption: Modules\n\n"
        for full_name, name in modules:
            rst_content += f"   {full_name}\n"
        rst_content += "\n"
    
    # Write the RST file
    rst_file = output_dir / f"{package_name}.rst"
    rst_file.write_text(rst_content)
    print(f"Generated: {rst_file}")

def generate_module_rst(module_name, output_dir):
    """Generate RST file for a module."""
    rst_content = f"""{module_name}
{'=' * len(module_name)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
   :imported-members:
   :special-members: __init__
   :exclude-members: __weakref__
"""
    
    # Write the RST file
    rst_file = output_dir / f"{module_name}.rst"
    rst_file.write_text(rst_content)
    print(f"Generated: {rst_file}")

def generate_main_index_rst(output_dir, project_root):
    """Generate the main index.rst with organized sections."""
    # Find all modules and subpackages
    fovi_path = project_root / "fovi"
    modules, subpackages = find_all_modules("fovi", str(fovi_path))
    
    # Organize modules by category
    core_modules = []
    utility_modules = []
    
    # Core modules (top-level important modules)
    core_module_names = ['fovi.saccadenet', 'fovi.trainer']
    for module_name, _ in modules:
        if module_name in core_module_names:
            core_modules.append(module_name)
    
    # Utility modules (other top-level modules)
    for module_name, _ in modules:
        if module_name not in core_module_names and not module_name.startswith('fovi.arch') and not module_name.startswith('fovi.sensing') and not module_name.startswith('fovi.utils'):
            utility_modules.append(module_name)
    
    # Build the RST content
    rst_content = """Welcome to fovi's documentation!
====================================

fovi is a PyTorch library for implementing foveated vision. This library provides tools for foveated sampling and foveated neural network architectures.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   read_me
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Core Components

   api/fovi.sensing
   api/fovi.arch
"""
    
    # Add core modules
    for module in sorted(core_modules):
        rst_content += f"   api/{module}\n"
    
    rst_content += """
.. toctree::
   :maxdepth: 2
   :caption: Utilities & Tools

   api/fovi.utils
"""
    
    # Add utility modules
    for module in sorted(utility_modules):
        rst_content += f"   api/{module}\n"
    
    rst_content += """

"""
    
    # Write the main index file to the docs directory
    index_file = output_dir / "index.rst"
    index_file.write_text(rst_content)
    print(f"Generated: {index_file}")

def generate_modules_rst(output_dir):
    """Generate the simple modules.rst file that clean_docs.sh removes."""
    rst_content = """fovi
=======

.. toctree::
   :maxdepth: 4

   fovi
"""
    
    # Write the modules.rst file
    modules_file = output_dir / "modules.rst"
    modules_file.write_text(rst_content)
    print(f"Generated: {modules_file}")

def convert_readme_to_rst(project_root, docs_dir):
    """Convert README.md to RST format using pandoc."""
    readme_md = project_root / "README.md"
    readme_rst = docs_dir / "read_me.rst"
    
    if not readme_md.exists():
        print(f"Warning: README.md not found at {readme_md}")
        return
    
    try:
        # Use pandoc to convert markdown to rst
        result = subprocess.run([
            'pandoc', 
            str(readme_md), 
            '-f', 'markdown', 
            '-t', 'rst', 
            '-o', str(readme_rst)
        ], capture_output=True, text=True, check=True)
        
        print(f"Converted README.md to RST: {readme_rst}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting README.md with pandoc: {e}")
        print(f"Pandoc stderr: {e.stderr}")
        # Fallback: create a simple include
        fallback_content = """README
======

.. include:: ../README.md
"""
        readme_rst.write_text(fallback_content)
        print(f"Created fallback read_me.rst with include directive")
        
    except FileNotFoundError:
        print("Warning: pandoc not found. Creating fallback read_me.rst with include directive")
        fallback_content = """README
======

.. include:: ../README.md
"""
        readme_rst.write_text(fallback_content)
        print(f"Created fallback read_me.rst with include directive")

def generate_api_index_rst(output_dir, project_root):
    """Generate the API index.rst with organized sections."""
    # Find all modules and subpackages
    fovi_path = project_root / "fovi"
    modules, subpackages = find_all_modules("fovi", str(fovi_path))
    
    # Organize modules by category
    core_modules = []
    utility_modules = []
    subpackage_modules = []
    
    # Core modules (top-level important modules)
    core_module_names = ['fovi.saccadenet', 'fovi.trainer']
    for module_name, _ in modules:
        if module_name in core_module_names:
            core_modules.append(module_name)
    
    # Utility modules (other top-level modules)
    for module_name, _ in modules:
        if module_name not in core_module_names and not module_name.startswith('fovi.arch') and not module_name.startswith('fovi.sensing') and not module_name.startswith('fovi.utils'):
            utility_modules.append(module_name)
    
    # Subpackage modules (modules within subpackages)
    for module_name, _ in modules:
        if module_name.startswith('fovi.arch.') or module_name.startswith('fovi.sensing.') or module_name.startswith('fovi.utils.'):
            subpackage_modules.append(module_name)
    
    # Build the RST content
    rst_content = """API Reference
=============

.. toctree::
   :maxdepth: 4
   :caption: Core API

   fovi
   fovi.sensing
   fovi.arch
"""
    
    # Add core modules
    for module in sorted(core_modules):
        rst_content += f"   {module}\n"
    
    rst_content += """
.. toctree::
   :maxdepth: 4
   :caption: Utilities & Tools

"""
    
    # Add utility modules
    for module in sorted(utility_modules):
        rst_content += f"   {module}\n"
    
    # Add subpackages section if there are any
    if subpackage_modules:
        rst_content += """
.. toctree::
   :maxdepth: 4
   :caption: Subpackages

"""
        # Add subpackage modules
        for module in sorted(subpackage_modules):
            rst_content += f"   {module}\n"
    
    rst_content += "\n"
    
    # Write the API index file
    api_index_file = output_dir / "index.rst"
    api_index_file.write_text(rst_content)
    print(f"Generated: {api_index_file}")

def main():
    """Main function to generate all RST files."""
    # Get the project root and docs directory
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    api_dir = docs_dir / "api"
    
    # Ensure api directory exists
    api_dir.mkdir(exist_ok=True)
    
    # Find the fovi package
    fovi_path = project_root / "fovi"
    if not fovi_path.exists():
        print("Error: fovi package not found!")
        return
    
    # Generate main index files
    generate_main_index_rst(docs_dir, project_root)
    generate_api_index_rst(api_dir, project_root)
    
    # Convert README.md to RST format using pandoc
    convert_readme_to_rst(project_root, docs_dir)
    
    # Generate the simple modules.rst file that clean_docs.sh removes
    generate_modules_rst(api_dir)
    
    # Generate RST for the main package
    generate_package_rst("fovi", str(fovi_path), api_dir)
    
    # Find all subpackages and modules
    modules, subpackages = find_all_modules("fovi", str(fovi_path))
    
    # Generate RST for all modules
    for module_name, _ in modules:
        generate_module_rst(module_name, api_dir)
    
    # Generate RST for all subpackages
    for package_name, _ in subpackages:
        package_path = fovi_path / package_name.split('.')[-1]
        generate_package_rst(package_name, str(package_path), api_dir)
    
    print("RST structure generation complete!")

if __name__ == "__main__":
    main() 