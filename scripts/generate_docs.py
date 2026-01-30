#!/usr/bin/env python3
"""
Script to generate documentation
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(command):
    """Run a command and return success status."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {result.stdout}")
    return True

def main():
    docs_dir = Path(__file__).parent.parent / "docs"
    build_dir = docs_dir / "_build"

    # Clean previous build
    print("🧹 Cleaning previous build...")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # Generate RST structure
    print("📝 Generating RST structure...")
    if not run_command("python scripts/generate_rst_structure.py"):
        print("⚠️  RST structure generation failed, but continuing...")

    # Move generated index.rst to docs directory if it exists in root
    root_index = Path(__file__).parent.parent / "index.rst"
    if root_index.exists():
        shutil.move(str(root_index), str(docs_dir / "index.rst"))
        print("📁 Moved index.rst to docs directory")

    # Copy notebooks into the api section so we can use them
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    api_dir = docs_dir / "api"
    if notebooks_dir.exists() and api_dir.exists():
        for nb_file in notebooks_dir.glob("*.ipynb"):
            dest = api_dir / nb_file.name
            shutil.copy2(nb_file, dest)
            print(f"📁 Copied {nb_file.name} to api directory")
    else:
        print("⚠️  Notebooks or api directory does not exist, skipping notebook copy.")

    # Change to docs directory
    os.chdir(docs_dir)

    # Generate autosummary files
    print("📝 Generating autosummary files...")
    if not run_command("sphinx-autogen -o api/_autosummary api/modules.rst"):
        print("⚠️  Autosummary generation failed, but continuing...")

    # Build the documentation
    print("🔨 Building documentation...")
    if not run_command("make html"):
        print("❌ Documentation build failed!")
        return False

    print("✅ Documentation generated successfully!")
    print(f"📖 Documentation available at: {build_dir / 'html' / 'index.html'}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 