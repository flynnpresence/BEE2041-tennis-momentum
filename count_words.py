import re

with open('blog.qmd', 'r') as f:
    text = f.read()

text = re.sub(r'^---.*?---', '', text, flags=re.DOTALL)
text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
text = re.sub(r'<[^>]+>', '', text)
text = re.sub(r'[#*\[\]()!`]', '', text)
text = re.sub(r'outputs/\S+', '', text)

words = text.split()
print(f'Word count: {len(words)}')
