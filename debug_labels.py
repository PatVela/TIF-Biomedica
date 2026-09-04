import sys
import glob
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src"
    )
)

from data2020 import (
    parse_header,
    load_scored_classes,
    dx_codes_to_label_vector
)


# -------------------------------------------------------------
# Cargar clases
# -------------------------------------------------------------

csv_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src",
    "dx_mapping_scored.csv"
)

classes, code_to_class_idx = load_scored_classes(
    csv_path
)

print("=" * 70)
print("DEBUG DE ETIQUETAS CINC2020")
print("=" * 70)

print()
print(f"Clases cargadas: {len(classes)}")

for i, cls in enumerate(classes):
    print(f"{i:02d}: {cls}")

# -------------------------------------------------------------
# Buscar headers
# -------------------------------------------------------------

hea_files = glob.glob(
    os.path.join(
        "dataset2020",
        "**",
        "*.hea"
    ),
    recursive=True
)

print()
print(f"Headers encontrados: {len(hea_files)}")

# -------------------------------------------------------------
# Inspeccionar 20 registros
# -------------------------------------------------------------

total_codes = 0
matched_codes = 0

for hea_path in hea_files[:20]:

    meta = parse_header(
        hea_path
    )

    print()
    print("-" * 70)
    print(
        f"Registro: {meta['record_name']}"
    )

    print(
        f"Base: {hea_path}"
    )

    print(
        f"Dx originales: {meta['dx_codes']}"
    )

    matched = []
    unmatched = []

    for code in meta['dx_codes']:

        total_codes += 1

        idx = code_to_class_idx.get(
            code
        )

        if idx is None:

            unmatched.append(
                code
            )

        else:

            matched_codes += 1

            matched.append(
                (
                    code,
                    idx,
                    classes[idx]
                )
            )

    print(
        f"Matched: {matched}"
    )

    print(
        f"Unmatched: {unmatched}"
    )

    y = dx_codes_to_label_vector(
        meta['dx_codes'],
        code_to_class_idx,
        len(classes)
    )

    print(
        f"Vector: {y.astype(int).tolist()}"
    )

print()
print("=" * 70)

print(
    f"Códigos totales: {total_codes}"
)

print(
    f"Códigos encontrados: {matched_codes}"
)

if total_codes > 0:

    print(
        f"Cobertura: "
        f"{100 * matched_codes / total_codes:.2f}%"
    )

print("=" * 70)