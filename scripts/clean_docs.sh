#!/bin/bash

# General-purpose script to clean up Sphinx documentation build artifacts
# This ensures a clean slate when regenerating documentation after refactoring
# 
# Usage: ./clean_docs.sh [docs_directory]
# If no directory is specified, defaults to 'docs' in the project root

set -e  # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 [docs_directory]"
    echo ""
    echo "Clean up Sphinx documentation build artifacts and old files."
    echo ""
    echo "Arguments:"
    echo "  docs_directory    Path to documentation directory (default: docs)"
    echo ""
    echo "Examples:"
    echo "  $0                # Clean up ./docs directory"
    echo "  $0 documentation  # Clean up ./documentation directory"
    echo "  $0 /path/to/docs  # Clean up specific directory"
    exit 1
}

# Parse command line arguments
if [ $# -gt 1 ]; then
    usage
fi

# Determine docs directory
if [ $# -eq 1 ]; then
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        usage
    fi
    DOCS_DIR="$1"
else
    # Default to 'docs' directory in project root
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    DOCS_DIR="$PROJECT_ROOT/docs"
fi

# Convert to absolute path
DOCS_DIR="$(cd "$DOCS_DIR" 2>/dev/null && pwd || echo "$DOCS_DIR")"

echo "🧹 Cleaning up Sphinx documentation build artifacts..."
echo "📁 Documentation directory: $DOCS_DIR"

# Check if docs directory exists
if [ ! -d "$DOCS_DIR" ]; then
    echo "❌ Error: Documentation directory '$DOCS_DIR' does not exist!"
    exit 1
fi

# Clean up Sphinx build artifacts
echo "🗑️  Removing Sphinx build directory..."
if [ -d "$DOCS_DIR/_build" ]; then
    rm -rf "$DOCS_DIR/_build"
    echo "   ✅ Removed _build directory"
else
    echo "   ℹ️  No _build directory found"
fi

# Clean up autosummary generated files
echo "🗑️  Removing autosummary generated files..."
if [ -d "$DOCS_DIR/api/_autosummary" ]; then
    rm -rf "$DOCS_DIR/api/_autosummary"
    echo "   ✅ Removed _autosummary directory"
else
    echo "   ℹ️  No _autosummary directory found"
fi

# Clean up old generated RST files (common patterns)
echo "🗑️  Removing old generated RST files..."

# Remove common auto-generated RST files that might be outdated
COMMON_GENERATED_PATTERNS=(
    "api/*.rst"
    "modules.rst"
    "api/modules.rst"
)

for pattern in "${COMMON_GENERATED_PATTERNS[@]}"; do
    find "$DOCS_DIR" -path "$DOCS_DIR/$pattern" -type f -delete 2>/dev/null || true
done

# Remove any RST files that look like they might be auto-generated
# (files with patterns like package.subpackage.module.rst)
echo "🗑️  Removing auto-generated module RST files..."
find "$DOCS_DIR" -name "*.rst" -type f | while read -r file; do
    filename=$(basename "$file")
    # Check if filename contains dots (likely auto-generated from module names)
    if [[ "$filename" == *.*.* ]]; then
        rm "$file"
        echo "   ✅ Removed $filename"
    fi
done

# Clean up Python cache files in docs
echo "🗑️  Removing Python cache files..."
find "$DOCS_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$DOCS_DIR" -name "*.pyc" -type f -delete 2>/dev/null || true
echo "   ✅ Removed Python cache files"

# Clean up any temporary files
echo "🗑️  Removing temporary files..."
find "$DOCS_DIR" -name "*.tmp" -type f -delete 2>/dev/null || true
find "$DOCS_DIR" -name "*.bak" -type f -delete 2>/dev/null || true
find "$DOCS_DIR" -name ".DS_Store" -type f -delete 2>/dev/null || true
echo "   ✅ Removed temporary files"

# Clean up any misplaced notebook files
echo "🗑️  Cleaning up misplaced notebook files..."
# Remove notebooks from api directory (they shouldn't be there)
if [ -d "$DOCS_DIR/api" ]; then
    find "$DOCS_DIR/api" -name "*.ipynb" -type f -delete 2>/dev/null || true
    echo "   ✅ Removed any misplaced notebook files from api directory"
fi

# Clean up any old autodoc cache
echo "🗑️  Cleaning up autodoc cache..."
if [ -f "$DOCS_DIR/conf.pyc" ]; then
    rm "$DOCS_DIR/conf.pyc"
    echo "   ✅ Removed conf.pyc"
fi

# Clean up any old environment files
echo "🗑️  Cleaning up old environment files..."
if [ -f "$DOCS_DIR/_build/doctrees/environment.pickle" ]; then
    rm -f "$DOCS_DIR/_build/doctrees/environment.pickle" 2>/dev/null || true
    echo "   ✅ Removed old environment.pickle"
fi

echo ""
echo "🎉 Documentation cleanup complete!"
echo ""
echo "📋 Summary of what was cleaned:"
echo "   • Sphinx build directory (_build/)"
echo "   • Autosummary generated files (_autosummary/)"
echo "   • Auto-generated RST files (package.module.rst patterns)"
echo "   • Python cache files (__pycache__, *.pyc)"
echo "   • Temporary files (*.tmp, *.bak, .DS_Store)"
echo "   • Misplaced notebook files"
echo "   • Old autodoc cache files"
echo ""
echo "✨ You can now regenerate your documentation with a clean slate!"
echo ""
echo "💡 Common next steps:"
echo "   • Generate new RST files: python scripts/generate_rst_structure.py"
echo "   • Build documentation: python -m sphinx -b html $DOCS_DIR $DOCS_DIR/_build/html"
echo "   • Or use your project's build script"
