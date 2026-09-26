"""
paridade.py
-----------
Garante que as mudanças no engine.py não alteram o que as IAs produzem.

Compara com as respostas da API ANTES das mudanças (JSONs de referência): diagnóstico,
janela, fatia central e ini_cranio idênticos; scores da CNN1 e predições da CNN2 iguais
dentro da tolerância numérica.

Uso (dentro do container da API):
    python scripts/paridade.py /data/AD_002_S_0619.nii /data/CN_002_S_0295.nii --referencia /out

Sai com código 1 se alguma verificação falhar.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402

TOL_SCORES_REL = 1e-5     # scores da CNN1 são regressão (dezenas): tolerância relativa
TOL_PREDICOES_ABS = 1e-5  # probabilidades (0–1): tolerância absoluta


def verificar(nome, ok, detalhe=""):
    print(f"  [{'OK' if ok else 'FALHA'}] {nome}{' — ' + detalhe if detalhe else ''}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arquivos", nargs="+")
    ap.add_argument("--referencia", required=True, help="pasta com <exame>.run1.json (API antes das mudanças)")
    args = ap.parse_args()

    import tensorflow as tf
    roi = tf.keras.models.load_model(os.getenv("MODEL_ROI_PATH", "models/modelo_cnn1_08_maio.h5"))
    clf = tf.keras.models.load_model(os.getenv("MODEL_CLF_PATH", "models/CNN-4-2023-1.h5"), compile=False)
    roi.predict_on_batch(np.zeros((engine.LOTE_CNN1, 176, 176, 3), np.float32))   # aquecimento, como na API
    clf.predict_on_batch(np.zeros((19, 176, 176, 3), np.float32))

    tudo_ok = True
    for caminho in args.arquivos:
        exame = os.path.basename(caminho).split(".")[0]
        ref = json.load(open(os.path.join(args.referencia, f"{exame}.run1.json")))
        print(f"\n{exame}", flush=True)

        t0 = time.perf_counter()
        novo = engine.run_pipeline(caminho, roi, clf, visualizacao=False)
        t1 = time.perf_counter()
        c1, r1 = novo["metadados_varredura_cnn1"], ref["metadados_varredura_cnn1"]
        s_novo, s_ref = np.array(c1["roi_scores"]), np.array(r1["roi_scores"])
        p_novo = np.array(novo["estatisticas_predicao"]["predicoes_19_slices"])
        p_ref = np.array(ref["estatisticas_predicao"]["predicoes_19_slices"])
        dif_s = float(np.max(np.abs(s_novo - s_ref) / np.maximum(np.abs(s_ref), 1e-6)))
        dif_p = float(np.max(np.abs(p_novo - p_ref)))
        print(" pipeline atual, agora em lotes:")
        tudo_ok &= verificar("diagnóstico", novo["diagnostico"] == ref["diagnostico"], novo["diagnostico"])
        tudo_ok &= verificar("janela e fatia central",
                             (c1["limite_inferior"], c1["limite_superior"], c1["slice_central"]) ==
                             (r1["limite_inferior"], r1["limite_superior"], r1["slice_central"]),
                             f"[{c1['limite_inferior']}, {c1['limite_superior']}] → {c1['slice_central']}")
        tudo_ok &= verificar("ini_cranio", novo["metadados_extracao_cnn2"]["inicio_cranio_y"] ==
                             ref["metadados_extracao_cnn2"]["inicio_cranio_y"])
        tudo_ok &= verificar("scores da CNN1", dif_s <= TOL_SCORES_REL, f"maior diferença relativa {dif_s:.2e}")
        tudo_ok &= verificar("predições da CNN2", dif_p <= TOL_PREDICOES_ABS, f"maior diferença absoluta {dif_p:.2e}")
        print(f"  tempo: {t1 - t0:.1f} s (etapas {novo['tempos_ms']})")

    print("\nRESULTADO:", "PARIDADE OK" if tudo_ok else "FALHOU — não publicar", flush=True)
    sys.exit(0 if tudo_ok else 1)


if __name__ == "__main__":
    main()
