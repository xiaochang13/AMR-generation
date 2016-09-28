
count = 0
for line in open('amr','rU'):
    line = line.rstrip()
    if line.startswith('#'):
        continue
    if line.startswith('('):
        count += 1
    print line

print count
