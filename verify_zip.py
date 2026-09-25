import zipfile, sys
zp = r'E:\aic\final-ihc\submissions\fullplus_cd68_v1.zip'
with zipfile.ZipFile(zp) as z:
    names = z.namelist()
    print(f"Total entries: {len(names)}")
    print(f"First 5: {names[:5]}")
    print(f"Last 5: {names[-5:]}")
    by_marker = {}
    for n in names:
        if '/' in n:
            marker = n.split('/')[1]
            by_marker[marker] = by_marker.get(marker, 0) + 1
    print(f"By marker: {by_marker}")
