#!/usr/bin/env python3
"""Numerischer Selbsttest des kvq-Patches -- OHNE Serverstart.

Warum ueberhaupt: ein vLLM-Start mit diesem Modell kostet 10-20 Minuten. Wenn
der Dequant-Zweig im Triton-Kernel falsch waere, saehe man das erst danach -- und
zwar als schlechte Qualitaetszahlen, also genau in der Form, in der man es am
leichtesten fehlinterpretiert ("fp8 kostet halt Qualitaet"). Dieser Test trennt
die beiden Fragen vorher sauber: rechnet der Kernel richtig (hier), und was
kostet die Quantisierung (Gates G2-G4).

Aufbau (Formen wie im echten Modell: 24 Q-Heads, 2 KV-Heads, head_dim 256,
page_size 64, Auswahlbreite 2051 = indexer_budget 2048 + compress_ratio 4 - 1):

  1. bf16-K/V-Seiten zufaellig fuellen, Auswahl + Seitentabelle bauen
  2. REFERENZ in float64 mit reinem torch rechnen (gather + softmax)
  3. Kernel im bf16-Pfad laufen lassen        -> Grundrauschen des Kernels
  4. K/V nach fp8-e4m3 quantisieren (Skala 1,0), zurueck nach bf16 lesen
     -> das ist exakt das, was der Kernel intern sehen wird
  5. REFERENZ auf den DEQUANTISIERTEN Werten rechnen
  6. Kernel im fp8-Pfad laufen lassen und gegen (5) vergleichen

Der entscheidende Vergleich ist 6 gegen 5: er misst NUR den Kernel, nicht die
Quantisierung. Stimmt er, ist der Dequant-Zweig korrekt -- unabhaengig davon,
wie gross der Quantisierungsfehler ausfaellt. Der Vergleich 6 gegen 2 zeigt
zusaetzlich, was fp8 an dieser Stelle real kostet.

Zusatzprobe: der bf16-Pfad muss nach dem Patch BITWEISE unveraendert sein
(USE_FP8=False kompiliert den Zweig weg). Dafuer laeuft Schritt 3 zweimal --
einmal ohne Skalen-Argumente, einmal mit -- und beide Ergebnisse muessen exakt
gleich sein.

Aufruf im Container:  python3 /qfn/kvq-kerneltest.py
"""

import sys

import torch

# model zuerst -- qsa.py direkt zu importieren loest einen (vorbestehenden)
# Zirkelimport aus.
from vllm.models.qwen3_8_flash_next.nvidia import model as _model  # noqa: F401
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention

NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
PAGE_SIZE = 64
TOPK = 2051
NUM_ROWS = 5  # Query-Zeilen; ungerade, damit kein Sonderfall "genau ein Tile"
NUM_REQ = 2
SEQ_LEN = 4096  # logische Kontextlaenge je Request
SEED = 0


def referenz(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
) -> torch.Tensor:
    """Dieselbe Rechnung wie der Kernel, aber in float64 mit reinem torch."""
    rows, heads, dim = q.shape
    group = heads // k_cache.shape[2]
    out = torch.zeros(rows, heads, dim, dtype=torch.float64, device=q.device)
    max_pages = block_table.shape[1]
    for r in range(rows):
        req = int(token_to_req[r])
        tok = indices[r]
        gueltig = tok >= 0
        tok = tok.clamp(min=0)
        seite = tok // PAGE_SIZE
        gueltig &= seite < max_pages
        versatz = tok % PAGE_SIZE
        phys = block_table[req, seite.clamp(max=max_pages - 1)]
        gueltig &= (phys >= 0) & (phys < k_cache.shape[0])
        idx = gueltig.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        p = phys[idx].long()
        o = versatz[idx].long()
        for kvh in range(k_cache.shape[2]):
            keys = k_cache[p, o, kvh].to(torch.float64)  # [n, dim]
            values = v_cache[p, o, kvh].to(torch.float64)
            for h in range(kvh * group, (kvh + 1) * group):
                scores = (q[r, h].to(torch.float64) @ keys.T) * (dim**-0.5)
                w = torch.softmax(scores, dim=0)
                out[r, h] = w @ values
    return out


def bericht(name: str, a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.to(torch.float64) - b.to(torch.float64)).abs()
    n = b.to(torch.float64).abs().mean().clamp(min=1e-12)
    print(
        f"  {name:<44} max {d.max().item():.3e}  "
        f"mittel {d.mean().item():.3e}  rel {(d.mean() / n).item():.3e}"
    )
    return (d.mean() / n).item()


def main() -> int:
    if not torch.cuda.is_available():
        print("FEHLER: keine CUDA-GPU sichtbar", file=sys.stderr)
        return 1
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name(0)}  SM{cap[0]}{cap[1]}")
    torch.manual_seed(SEED)

    pages_pro_req = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = NUM_REQ * pages_pro_req + 3  # +3 = ein paar unbenutzte Seiten

    # K/V bewusst klein halten (Post-RMSNorm-Groessenordnung); fp8-e4m3 hat
    # Reichweite +-448, die Skala 1,0 ist damit unkritisch -- genau das ist die
    # Annahme, die Stufe 1 traegt, und sie wird hier mitgemessen.
    q = (torch.randn(NUM_ROWS, NUM_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
    k_cache = (
        torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5
    ).bfloat16()
    v_cache = (
        torch.randn(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5
    ).bfloat16()

    block_table = torch.arange(
        NUM_REQ * pages_pro_req, dtype=torch.int32, device=dev
    ).view(NUM_REQ, pages_pro_req)

    token_to_req = torch.randint(
        0, NUM_REQ, (NUM_ROWS,), dtype=torch.int32, device=dev
    )
    # Auswahl: echte Token + bewusst ein Teil ungueltig (-1), wie im Betrieb,
    # wenn der Indexer weniger als die volle Breite fuellt.
    indices = torch.randint(
        0, SEQ_LEN, (NUM_ROWS, TOPK), dtype=torch.int32, device=dev
    )
    indices[:, TOPK // 2 :] = -1

    print("\n[1] Referenz bf16 (float64-Rechnung)")
    ref_bf16 = referenz(q, k_cache, v_cache, indices, block_table, token_to_req)

    print("[2] Kernel bf16-Pfad, ohne Skalen-Argumente")
    out_a = qsa_sparse_paged_attention(
        q, k_cache, v_cache, indices, block_table, token_to_req
    )
    print("[3] Kernel bf16-Pfad, MIT Skalen-Argumenten (muessen ignoriert werden)")
    eins = torch.tensor(1.0, dtype=torch.float32, device=dev)
    out_b = qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        indices,
        block_table,
        token_to_req,
        k_scale=eins,
        v_scale=eins,
    )
    bitgleich = torch.equal(out_a, out_b)
    print(f"  bf16-Pfad bitweise identisch mit/ohne Skalen: {bitgleich}")

    print("\n[4] fp8-e4m3-Quantisierung mit Skala 1,0")
    # Bewusst ueber uint8 UND ZURUECK: genau so liegt der Cache im Betrieb.
    # vLLM allokiert quantisierten KV als torch.uint8, die e4m3-Bytes sind also
    # nur "getarnt"; forward_qsa deutet sie per view() zurueck. Wenn diese
    # Umdeutung irgendwo eine Kopie erzwaenge oder an Schrittweiten scheiterte,
    # muss es HIER auffallen und nicht erst nach 15 Minuten Serverstart.
    k_fp8 = k_cache.to(torch.float8_e4m3fn).view(torch.uint8).view(torch.float8_e4m3fn)
    v_fp8 = v_cache.to(torch.float8_e4m3fn).view(torch.uint8).view(torch.float8_e4m3fn)
    k_deq = k_fp8.to(torch.bfloat16)
    v_deq = v_fp8.to(torch.bfloat16)
    saettigung = (k_cache.abs() > 448).float().mean().item()
    print(
        f"  K/V ausserhalb der e4m3-Reichweite (+-448): {saettigung * 100:.4f} %"
        "  (0 % = Skala 1,0 traegt)"
    )

    print("[5] Referenz auf den DEQUANTISIERTEN Werten (float64)")
    ref_fp8 = referenz(q, k_deq, v_deq, indices, block_table, token_to_req)

    print("[6] Kernel fp8-Pfad")
    out_fp8 = qsa_sparse_paged_attention(
        q,
        k_fp8,
        v_fp8,
        indices,
        block_table,
        token_to_req,
        k_scale=eins,
        v_scale=eins,
    )

    print("\nVergleiche:")
    r_bf16 = bericht("[2] Kernel bf16   gegen [1] Referenz bf16", out_a, ref_bf16)
    r_kern = bericht("[6] Kernel fp8    gegen [5] Referenz fp8 ", out_fp8, ref_fp8)
    r_quant = bericht("[6] Kernel fp8    gegen [1] Referenz bf16", out_fp8, ref_bf16)

    print("\nDeutung:")
    print(
        "  Zeile 2 ist das Mass fuer den KERNEL (Quantisierung herausgerechnet).\n"
        "  Sie muss in derselben Groessenordnung liegen wie Zeile 1 -- dann rechnet\n"
        "  der Dequant-Zweig richtig. Zeile 3 ist der Preis der Quantisierung."
    )

    # ---------------------------------------------------------------------
    # Kachel-Profil-Durchlauf.
    #
    # WARUM DAS HIER STEHT: die erste Fassung dieses Tests lief nur mit 5
    # Query-Zeilen. Der Wrapper waehlt seine Kachelgroesse aber nach der Zahl der
    # Programme (Zeilen x KV-Koepfe) -- 5 Zeilen landen bei BLOCK_N=16. Der
    # Prefill-Fall BLOCK_N=64 wurde nie angefasst, und GENAU DER kippte dann im
    # Server mit "triton OutOfResources: shared memory, Required 106496,
    # Hardware limit 101376". Ein Test, der nur einen Zweig anfasst, belegt nur
    # diesen Zweig. Also: alle fuenf Profile des Wrappers ablaufen.
    #
    # Geprueft wird hier nicht mehr gegen float64 (zu teuer bei 300 Zeilen),
    # sondern Kernel-gegen-Kernel: derselbe Kernel auf DEQUANTISIERTEN bf16-K/V
    # gegen den fp8-Zweig. Beide muessen dasselbe rechnen -- und beide muessen
    # ueberhaupt starten, was der eigentliche Punkt ist.
    print("\n[7] Kachel-Profile des Wrappers (Ressourcen + Uebereinstimmung)")
    print("      Zeilen  Programme  BLOCK_N  Stufen   max|fp8 - bf16(deq)|")
    profil_fehler = 0
    for rows in (2, 5, 64, 200, 300):
        programme = rows * NUM_KV_HEADS
        gruppe = NUM_HEADS // NUM_KV_HEADS
        bm = 1 << (gruppe - 1).bit_length()
        grenze = 8 if bm <= 8 else 4
        if programme <= grenze:
            bn = 16
        elif programme < 32:
            bn = 16
        else:
            bn = 64
        stufen = 1 if bn > 16 else 2
        qp = (torch.randn(rows, NUM_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
        ip = torch.randint(0, SEQ_LEN, (rows, TOPK), dtype=torch.int32, device=dev)
        ip[:, TOPK // 2 :] = -1
        tp = torch.randint(0, NUM_REQ, (rows,), dtype=torch.int32, device=dev)
        try:
            o_deq = qsa_sparse_paged_attention(qp, k_deq, v_deq, ip, block_table, tp)
            o_fp8 = qsa_sparse_paged_attention(
                qp, k_fp8, v_fp8, ip, block_table, tp, k_scale=eins, v_scale=eins
            )
        except Exception as e:  # noqa: BLE001
            print(f"      {rows:>6}  {programme:>9}  {bn:>7}  {stufen:>6}   FEHLER: {e}")
            profil_fehler += 1
            continue
        d = (o_fp8.to(torch.float64) - o_deq.to(torch.float64)).abs().max().item()
        marke = "ok" if d < 5e-3 else "ABWEICHUNG"
        print(f"      {rows:>6}  {programme:>9}  {bn:>7}  {stufen:>6}   {d:.3e}  {marke}")
        if d >= 5e-3:
            profil_fehler += 1

    fehler = 0
    if profil_fehler:
        print(f"\nFEHLGESCHLAGEN: {profil_fehler} Kachel-Profil(e) defekt.")
        fehler += profil_fehler
    if not bitgleich:
        print("\nFEHLGESCHLAGEN: bf16-Pfad ist nicht mehr bitweise stabil.")
        fehler += 1
    # Der Kernelfehler im fp8-Pfad darf das bf16-Grundrauschen nur um einen
    # kleinen Faktor uebersteigen (bf16-Rundung der dequantisierten Werte).
    if r_kern > max(5 * r_bf16, 5e-3):
        print(
            f"\nFEHLGESCHLAGEN: Kernelfehler im fp8-Pfad {r_kern:.3e} zu gross "
            f"(bf16-Grundrauschen {r_bf16:.3e})."
        )
        fehler += 1
    if r_quant > 0.05:
        print(
            f"\nWARNUNG: Quantisierungsfehler {r_quant:.3e} ungewoehnlich gross "
            "-- Skalen-Kalibrierung pruefen."
        )

    if fehler:
        return 3
    print("\nBESTANDEN: Dequant-Zweig korrekt, bf16-Pfad unveraendert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
