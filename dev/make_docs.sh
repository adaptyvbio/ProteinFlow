python dev/update_init_docs.py
rm -r docs/*
pdoc3 -o docs --html --force --template-dir pdoc/templates proteinflow 
mv docs/proteinflow/* docs/
rm -r docs/proteinflow
cp media/adaptyv_logo.png docs/
# Copy codeboarding HTML files to docs directory if they exist
if [ -d ".codeboarding" ]; then
  cp -r .codeboarding/*.html docs/ 2>/dev/null || true
fi

# Generate integrated versions of standalone HTML files
python dev/integrate_html_files.py