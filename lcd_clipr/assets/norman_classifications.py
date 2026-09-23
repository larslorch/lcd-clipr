norman_classifications = {
    "neomorphism": [
        ("CBL", "TGFBR2"),
        ("KLF1", "TGFBR2"),
        ("MAP2K6", "SPI1"),
        ("SAMD1", "TGFBR2"),
        ("TGFBR2", "CBARP"),
        ("TGFBR2", "ETS2"),
        ("CBL", "UBASH3A"),
        ("CEBPE", "KLF1"),
        ("DUSP9", "MAPK1"),
        ("FOSB", "PTPN12"),
        ("PLK4", "STIL"),
        ("PTPN12", "OSR2"),
        ("ZC3HAV1", "CEBPE")
    ],
    "additive": [
        ("BPGM", "SAMD1"),
        ("CEBPB", "MAPK1"),
        ("CEBPB", "OSR2"),
        ("DUSP9", "PRTG"),
        ("FOSB", "OSR2"),
        ("IRF1", "SET"),
        ("MAP2K3", "ELMSAN1"),
        ("MAP2K6", "ELMSAN1"),
        ("POU3F2", "FOXL2"),
        ("RHOXF2B", "SET"),
        ("SAMD1", "PTPN12"),
        ("SAMD1", "UBASH3B"),
        ("SAMD1", "ZBTB1"),
        ("SGK1", "TBX2"),
        ("TBX3", "TBX2"),
        ("ZBTB10", "SNAI1")
    ],
    "epistasis": [
        ("AHR", "KLF1"),
        ("MAPK1", "TGFBR2"),
        ("TGFBR2", "IGDCC3"),
        ("TGFBR2", "PRTG"),
        ("UBASH3B", "OSR2"),
        ("DUSP9", "ETS2"),
        ("KLF1", "CEBPA"),
        ("MAP2K6", "IKZF3"),
        ("ZC3HAV1", "CEBPA")
    ],
    "redundancy": [
        ("CDKN1C", "CDKN1A"),
        ("MAP2K3", "MAP2K6"),
        ("CEBPB", "CEBPA"),
        ("CEBPE", "CEBPA"),
        ("CEBPE", "SPI1"),
        ("ETS2", "MAPK1"),
        ("FOSB", "CEBPE"),
        ("FOXA3", "FOXA1")
    ],
    "potentiation": [
        ("CNN1", "UBASH3A"),
        ("ETS2", "MAP7D1"),
        ("FEV", "CBFA2T3"),
        ("FEV", "ISL2"),
        ("FEV", "MAP7D1"),
        ("PTPN12", "UBASH3A")
    ],
    "strong-synergy-similar": [
        ("CBL", "CNN1"),
        ("CBL", "PTPN12"),
        ("CBL", "PTPN9"),
        ("CBL", "UBASH3B"),
        ("FOXA3", "FOXL2"),
        ("FOXA3", "HOXB9"),
        ("FOXL2", "HOXB9"),
        ("UBASH3B", "CNN1"),
        ("UBASH3B", "PTPN12"),
        ("UBASH3B", "PTPN9"),
        ("UBASH3B", "ZBTB25")
    ],
    "suppression": [
        ("CEBPB", "PTPN12"),
        ("CEBPE", "CNN1"),
        ("CEBPE", "PTPN12"),
        ("CNN1", "MAPK1"),
        ("ETS2", "CNN1"),
        ("ETS2", "IGDCC3"),
        ("ETS2", "PRTG"),
        ("FOSB", "UBASH3B"),
        ("IGDCC3", "MAPK1"),
        ("LYL1", "CEBPB"),
        ("MAPK1", "PRTG"),
        ("PTPN12", "SNAI1")
    ],
    "strong-synergy-dissimilar": [
        ("AHR", "FEV"),
        ("DUSP9", "SNAI1"),
        ("FOXA1", "FOXF1"),
        ("FOXA1", "FOXL2"),
        ("FOXA1", "HOXB9"),
        ("FOXF1", "FOXL2"),
        ("FOXF1", "HOXB9"),
        ("FOXL2", "MEIS1"),
        ("IGDCC3", "ZBTB25"),
        ("POU3F2", "CBFA2T3"),
        ("PTPN12", "ZBTB25"),
        ("SNAI1", "DLX2"),
        ("SNAI1", "UBASH3B")
    ]
}

gears_classifications = dict()
gears_classifications["synergy"] = []
for key in norman_classifications.keys():
    if key in ["potentiation", "strong-synergy-similar", "strong-synergy-dissimilar"]:
        gears_classifications["synergy"] += norman_classifications[key]
    else:
        gears_classifications[key] = norman_classifications[key]
