pdoc3 -o docs --html --force --template-dir pdoc/templates proteinflow 
mv docs/proteinflow/* docs/
rm -r docs/proteinflow