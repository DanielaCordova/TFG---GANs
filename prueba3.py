fmap_base=8192
fmap_decay=1.0
fmap_max=512

def nf(stage):
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

for i in range(6,1,-1):
    print("res = " + str(i) + " nf = " + str(nf(i-1)) + " nf_i = " + str(nf(i)))
    print("      " + str(fmap_base) + " /  2 ** " + str(i-1) + " * " + str(fmap_decay))
    print("      " + str(fmap_base) + " /  2 ** " + str((i-1) * fmap_decay))
    print("      " + str(fmap_base) + " / " + str(2**((i-1)*fmap_decay)))
