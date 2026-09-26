"""
gerar_miniaturas.py
-------------------
Gera a miniatura (WebP, fatia coronal do meio, convenção neurológica) de cada exame
listado em <AMOSTRAS_DIR>/amostras.json e grava o campo "miniatura" no manifesto.

Uso:
    python scripts/gerar_miniaturas.py [pasta_das_amostras]
"""

import json
import os
import sys

import cv2
import nibabel as nib
import numpy as np

LADO = 160


def miniatura(caminho_nifti: str) -> np.ndarray:
    vol = nib.as_closest_canonical(nib.load(caminho_nifti)).get_fdata()
    fatia = vol[:, vol.shape[1] // 2, :]            # linhas = x (E→D), colunas = z (I→S)
    img = np.flipud(fatia.T)                          # superior no topo, esquerda do paciente à esquerda
    nz = img[img > 0]
    p2, p98 = np.percentile(nz, [2, 98]) if nz.size else (0, 1)
    u8 = (np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1) * 255).astype(np.uint8)
    h, w = u8.shape
    esc = LADO / max(h, w)
    u8 = cv2.resize(u8, (max(1, round(w * esc)), max(1, round(h * esc))), interpolation=cv2.INTER_AREA)
    quadro = np.zeros((LADO, LADO), dtype=np.uint8)
    y, x = (LADO - u8.shape[0]) // 2, (LADO - u8.shape[1]) // 2
    quadro[y:y + u8.shape[0], x:x + u8.shape[1]] = u8
    return quadro


def main():
    pasta = sys.argv[1] if len(sys.argv) > 1 else os.getenv("AMOSTRAS_DIR", "amostras")
    manifesto = os.path.join(pasta, "amostras.json")
    dados = json.load(open(manifesto, encoding="utf-8"))
    os.makedirs(os.path.join(pasta, "miniaturas"), exist_ok=True)
    for a in dados["amostras"]:
        destino = os.path.join("miniaturas", f"{a['id']}.webp")
        cv2.imwrite(os.path.join(pasta, destino), miniatura(os.path.join(pasta, a["arquivo"])),
                    [cv2.IMWRITE_WEBP_QUALITY, 85])
        a["miniatura"] = destino
        print("ok:", a["id"], "->", destino)
    with open(manifesto, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
