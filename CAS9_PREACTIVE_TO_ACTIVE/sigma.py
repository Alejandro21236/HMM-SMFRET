import parmed as pmd

structure = pmd.load_file("first_frame.pdb")

seen = set()
for res in structure.residues:
    key = (res.idx, res.name)
    if key not in seen:
        seen.add(key)
        print(f"{res.idx} {res.name}")
