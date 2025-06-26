#!/usr/bin/env python3
"""
Script to integrate standalone HTML files into the pdoc template structure.

This script generates wrapper pages that maintain the sidebar navigation.
"""

import os
import re
from pathlib import Path


def extract_html_content(filepath):
    """Extract content and metadata from HTML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Documentation"
        
        # Extract body content
        body_match = re.search(r'<body[^>]*>(.*?)</body>', content, re.DOTALL | re.IGNORECASE)
        if body_match:
            body_content = body_match.group(1).strip()
        else:
            # If no body tags, use everything after head
            head_end = content.find('</head>')
            if head_end != -1:
                body_content = content[head_end + 7:].strip()
                # Remove html and body tags if present
                body_content = re.sub(r'</?html[^>]*>', '', body_content, flags=re.IGNORECASE)
                body_content = re.sub(r'</?body[^>]*>', '', body_content, flags=re.IGNORECASE)
            else:
                body_content = content
        
        # Extract custom styles from the head section
        style_pattern = r'<style[^>]*>(.*?)</style>'
        style_matches = re.findall(style_pattern, content, re.DOTALL | re.IGNORECASE)
        
        # Filter out potentially conflicting styles but keep useful ones
        useful_styles = []
        for style in style_matches:
            # Keep styles that seem specific to content rather than layout
            if any(keyword in style.lower() for keyword in ['badge', 'mermaid', 'highlight', 'code', 'pre', 'table', 'img']):
                useful_styles.append(style.strip())
        
        styles = '\n'.join(useful_styles) if useful_styles else ''
        
        # Extract any scripts that might be needed (like mermaid)
        script_pattern = r'<script[^>]*(?:src=["\'][^"\']*["\']|type=["\'][^"\']*["\'])*[^>]*>.*?</script>'
        script_matches = re.findall(script_pattern, content, re.DOTALL | re.IGNORECASE)
        scripts = '\n'.join(script_matches) if script_matches else ''
        
        # Also extract script tags without closing tags (like imports)
        import_script_matches = re.findall(r'<script[^>]*type=["\']module["\'][^>]*>.*?</script>', content, re.DOTALL | re.IGNORECASE)
        if import_script_matches:
            scripts = '\n'.join(import_script_matches) + '\n' + scripts
        
        return {
            'title': title,
            'content': body_content,
            'styles': styles,
            'scripts': scripts
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def generate_pdoc_template():
    """Get the base pdoc template structure from existing docs."""
    index_path = "docs/index.html"
    if not os.path.exists(index_path):
        return None
    
    with open(index_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the base template structure
    # We'll use everything up to the main content section
    content_start = content.find('<article id="content">')
    if content_start == -1:
        return None
    
    before_content = content[:content_start]
    
    # Find the sidebar
    sidebar_start = content.find('<nav id="sidebar">')
    sidebar_end = content.find('</nav>', sidebar_start) + 6
    if sidebar_start == -1 or sidebar_end == -1:
        return None

    sidebar = content[sidebar_start:sidebar_end]

    # Find the footer
    footer_start = content.find('<footer id="footer">')
    footer_end = content.find('</html>')
    if footer_start == -1 or footer_end == -1:
        return None

    footer = content[footer_start:footer_end]

    return {
        'before_content': before_content,
        'sidebar': sidebar,
        'footer': footer
    }


def create_integrated_page(html_data, template_data, output_path, standalone_files, page_type="guide"):
    """Create an integrated page with pdoc template structure."""
    # The integration script only handles wrapping content in pdoc structure
    # All navigation (Additional Resources, Codeboardings) is managed by html.mako template
    
    # Insert content into sidebar
    sidebar = template_data['sidebar']
    
    # Fix all links in the sidebar that start with # to point to index.html#
    # This ensures navigation back to the main index page works correctly
    import re
    sidebar = re.sub(r'href="#([^"]+)"', r'href="index.html#\1"', sidebar)
    
    integrated_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />
  <meta name="generator" content="pdoc 0.11.6" />
  <title>{html_data['title']}</title>
  <meta name="description" content="{html_data['title']}" />
  <link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/sanitize.min.css" integrity="sha256-PK9q560IAAa6WVRRh76LtCaI8pjTJ2z11v0miyNNjrs=" crossorigin>
  <link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/typography.min.css" integrity="sha256-7l/o7C8jubJiy74VsKTidCy1yBkRtiUGbVkYBylBqUg=" crossorigin>
  <link rel="stylesheet preload" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/styles/github.min.css" crossorigin>
  
  <!-- Pdoc Styles -->
  <style>:root{{--highlight-color:#fe9}}.flex{{display:flex !important}}body{{line-height:1.5em}}#content{{padding:20px}}#sidebar{{padding:30px;overflow:hidden}}#sidebar > *:last-child{{margin-bottom:2cm}}.http-server-breadcrumbs{{font-size:130%;margin:0 0 15px 0}}#footer{{font-size:.75em;padding:5px 30px;border-top:1px solid #ddd;text-align:right}}#footer p{{margin:0 0 0 1em;display:inline-block}}#footer p:last-child{{margin-right:30px}}h1,h2,h3,h4,h5{{font-weight:300}}h1{{font-size:2.5em;line-height:1.1em}}h2{{font-size:1.75em;margin:1em 0 .50em 0}}h3{{font-size:1.4em;margin:25px 0 10px 0}}h4{{margin:0;font-size:105%}}h1:target,h2:target,h3:target,h4:target,h5:target,h6:target{{background:var(--highlight-color);padding:.2em 0}}a{{color:#058;text-decoration:none;transition:color .3s ease-in-out}}a:hover{{color:#e82}}.title code{{font-weight:bold}}h2[id^="header-"]{{margin-top:2em}}.ident{{color:#900}}pre code{{background:#f8f8f8;font-size:.8em;line-height:1.4em}}code{{background:#f2f2f1;padding:1px 4px;overflow-wrap:break-word}}h1 code{{background:transparent}}pre{{background:#f8f8f8;border:0;border-top:1px solid #ccc;border-bottom:1px solid #ccc;margin:1em 0;padding:1ex}}#http-server-module-list{{display:flex;flex-flow:column}}#http-server-module-list div{{display:flex}}#http-server-module-list dt{{min-width:10%}}#http-server-module-list p{{margin-top:0}}.toc ul,#index{{list-style-type:none;margin:0;padding:0}}#index code{{background:transparent}}#index h3{{border-bottom:1px solid #ddd}}#index ul{{padding:0}}#index h4{{margin-top:.6em;font-weight:bold}}@media (min-width:200ex){{#index .two-column{{column-count:2}}}}@media (min-width:300ex){{#index .two-column{{column-count:3}}}}dl{{margin-bottom:2em}}dl dl:last-child{{margin-bottom:4em}}dd{{margin:0 0 1em 3em}}#header-classes + dl > dd{{margin-bottom:3em}}dd dd{{margin-left:2em}}dd p{{margin:10px 0}}.name{{background:#eee;font-weight:bold;font-size:.85em;padding:5px 10px;display:inline-block;min-width:40%}}.name:hover{{background:#e0e0e0}}dt:target .name{{background:var(--highlight-color)}}.name > span:first-child{{white-space:nowrap}}.name.class > span:nth-child(2){{margin-left:.4em}}.inherited{{color:#999;border-left:5px solid #eee;padding-left:1em}}.inheritance em{{font-style:normal;font-weight:bold}}.desc h2{{font-weight:400;font-size:1.25em}}.desc h3{{font-size:1em}}.desc dt code{{background:inherit}}.source summary,.git-link-div{{color:#666;text-align:right;font-weight:400;font-size:.8em;text-transform:uppercase}}.source summary > *{{white-space:nowrap;cursor:pointer}}.git-link{{color:inherit;margin-left:1em}}.source pre{{max-height:500px;overflow:auto;margin:0}}.source pre code{{font-size:12px;overflow:visible}}.hlist{{list-style:none}}.hlist li{{display:inline}}.hlist li:after{{content:',\\2002'}}.hlist li:last-child:after{{content:none}}.hlist .hlist{{display:inline;padding-left:1em}}img{{max-width:100%}}td{{padding:0 .5em}}.admonition{{padding:.1em .5em;margin-bottom:1em}}.admonition-title{{font-weight:bold}}.admonition.note,.admonition.info,.admonition.important{{background:#aef}}.admonition.todo,.admonition.versionadded,.admonition.tip,.admonition.hint{{background:#dfd}}.admonition.warning,.admonition.versionchanged,.admonition.deprecated{{background:#fd4}}.admonition.error,.admonition.danger,.admonition.caution{{background:lightpink}}</style>
  <style media="screen and (min-width: 700px)">@media screen and (min-width:700px){{#sidebar{{width:30%;height:100vh;overflow:auto;position:sticky;top:0}}#content{{width:70%;padding:3em 4em;border-left:1px solid #ddd}}pre code{{font-size:1em}}.item .name{{font-size:1em}}main{{display:flex;flex-direction:row-reverse;justify-content:flex-end}}.toc ul ul,#index ul{{padding-left:1.5em}}.toc > ul > li{{margin-top:.5em}}}}</style>
  <style media="print">@media print{{#sidebar h1{{page-break-before:always}}.source{{display:none}}}}@media print{{*{{background:transparent !important;color:#000 !important;box-shadow:none !important;text-shadow:none !important}}a[href]:after{{content:" (" attr(href) ")";font-size:90%}}a[href][title]:after{{content:none}}abbr[title]:after{{content:" (" attr(title) ")"}} .ir a:after,a[href^="javascript:"]:after,a[href^="#"]:after{{content:""}}pre,blockquote{{border:1px solid #999;page-break-inside:avoid}}thead{{display:table-header-group}}tr,img{{page-break-inside:avoid}}img{{max-width:100% !important}}@page{{margin:0.5cm}}p,h2,h3{{orphans:3;widows:3}}h1,h2,h3,h4,h5,h6{{page-break-after:avoid}}}}</style>

  <!-- Additional styles for custom content -->
  <style>
    .badges img {{
      margin-right: 0.5rem;
      margin-bottom: 0.25rem;
    }}
    .mermaid {{
      text-align: center;
      margin: 1rem 0;
    }}
    #section-intro .badges {{
      margin-bottom: 1rem;
    }}
    /* Custom styles from original HTML */
    {html_data['styles']}
  </style>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/highlight.min.js" integrity="sha256-Uv3H6lx7dJmRfRvH8TH6kJD1TSK1aFcwgx+mdg3epi8=" crossorigin></script>
  <script>window.addEventListener('DOMContentLoaded', () => hljs.initHighlighting())</script>
  <link rel="shortcut icon" type="image/x-icon" href="adaptyv_logo.png?">
</head>
<body>
<main>
<article id="content">
<section id="section-intro">
{html_data['content']}
</section>
</article>

{sidebar}
</main>

{template_data['footer']}

{html_data['scripts']}
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(integrated_html)


def main():
    """Integrate HTML files into pdoc template structure."""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("docs directory not found")
        return
    
    # Get template structure
    template_data = generate_pdoc_template()
    if not template_data:
        print("Could not extract template structure from index.html")
        return
    
    # Find all standalone HTML files (exclude index.html and pdoc-generated files)
    # These are typically files that don't have the pdoc structure
    standalone_files = []
    exclude_patterns = ['index.html']  # Files to skip
    
    for html_file in docs_dir.glob('*.html'):
        filename = html_file.name
        if filename not in exclude_patterns:
            # Check if it's a standalone file by looking for pdoc structure
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                # If it doesn't have the pdoc sidebar, it's standalone
                if '<nav id="sidebar">' not in content:
                    standalone_files.append(filename)
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
    
    if not standalone_files:
        print("No standalone HTML files found to integrate")
        return
    
    print(f"Found {len(standalone_files)} standalone HTML files to integrate:")
    for filename in standalone_files:
        print(f"  - {filename}")
    
    # Process each standalone file
    for filename in standalone_files:
        filepath = docs_dir / filename
        if filepath.exists():
            print(f"\nIntegrating {filename}...")
            
            # Extract content from the HTML file
            html_data = extract_html_content(str(filepath))
            if html_data:
                # Create integrated version with sidebar navigation
                create_integrated_page(html_data, template_data, str(filepath), standalone_files)
                print(f"✓ Successfully integrated {filename}")
            else:
                print(f"✗ Failed to extract content from {filename}")
        else:
            print(f"File {filename} not found, skipping...")
    
    print(f"\nIntegration complete! Processed {len(standalone_files)} files.")


if __name__ == "__main__":
    main()
