import json

with open(r'd:\MCQ\website\data\COMS.json', 'r', encoding='utf-8') as f:
    q = json.load(f)

print(f'Total questions: {len(q)}')

# Simple exact/near-exact match check using first 100 chars
seen = {}
dups = 0
dup_list = []

for item in q:
    key = item['question'].lower().strip()[:100]
    if key in seen:
        dups += 1
        dup_list.append((seen[key], item['id'], key[:70]))
    else:
        seen[key] = item['id']

print(f'Duplicate pairs found: {dups}')
print()

if dup_list:
    print('Duplicates:')
    for id1, id2, text in dup_list[:10]:
        print(f'  #{id1} & #{id2}: {text}...')
    if len(dup_list) > 10:
        print(f'  ... and {len(dup_list) - 10} more')
else:
    print('✅ No duplicates found!')
