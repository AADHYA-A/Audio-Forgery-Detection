import csv

genuine = []
forged = []
speakers_g = set()
speakers_f = set()

with open('release_in_the_wild/meta.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['label'] == 'bona-fide' and row['speaker'] not in speakers_g and len(genuine) < 6:
            genuine.append(row)
            speakers_g.add(row['speaker'])
        elif row['label'] == 'spoof' and row['speaker'] not in speakers_f and len(forged) < 6:
            forged.append(row)
            speakers_f.add(row['speaker'])
        if len(genuine) >= 6 and len(forged) >= 6:
            break

print("GENUINE:")
for g in genuine:
    print(f"  {g['file']} | {g['speaker']} | {g['label']}")
print("FORGED:")
for f2 in forged:
    print(f"  {f2['file']} | {f2['speaker']} | {f2['label']}")
