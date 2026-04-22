import re

with open('blog.qmd', 'r') as f:
    text = f.read()

# Remove YAML header
text = re.sub(r'^---.*?---', '', text, flags=re.DOTALL)
# Remove raw HTML blocks
text = re.sub(r'```\{=html\}.*?```', '', text, flags=re.DOTALL)
# Remove all HTML tags
text = re.sub(r'<[^>]+>', '', text)
# Remove Quarto div markers
text = re.sub(r':::\s*\{[^}]*\}', '', text)
text = re.sub(r':::', '', text)
# Remove image markdown
text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
# Remove markdown headings
text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
# Remove button HTML remnants
text = re.sub(r'<button.*?</button>', '', text, flags=re.DOTALL)
# Remove remaining markdown syntax
text = re.sub(r'[*`]', '', text)

words = [w for w in text.split() if len(w) > 1]
print(f'Word count: {len(words)}')
