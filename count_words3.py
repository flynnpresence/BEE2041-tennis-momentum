import re

with open('blog.qmd', 'r') as f:
    lines = f.readlines()

prose_lines = []
in_yaml = False
in_html_block = False
in_code_block = False

for line in lines:
    stripped = line.strip()
    
    if stripped == '---' and not in_yaml:
        in_yaml = True
        continue
    if stripped == '---' and in_yaml:
        in_yaml = False
        continue
    if in_yaml:
        continue
    if '```{=html}' in stripped:
        in_html_block = True
        continue
    if in_html_block and stripped == '```':
        in_html_block = False
        continue
    if in_html_block:
        continue
    if stripped.startswith('```'):
        in_code_block = not in_code_block
        continue
    if in_code_block:
        continue
    if stripped.startswith('#'):
        continue
    if stripped.startswith(':::'):
        continue
    if stripped.startswith('!'):
        continue
    if stripped.startswith('<'):
        continue
    if stripped.startswith(':'):
        continue
    if not stripped:
        continue
    
    prose_lines.append(stripped)

text = ' '.join(prose_lines)
text = re.sub(r'<[^>]+>', '', text)
text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
words = text.split()
print(f'Prose word count: {len(words)}')
print(f'First 5 lines counted:')
for l in prose_lines[:5]:
    print(f'  {l[:80]}')
