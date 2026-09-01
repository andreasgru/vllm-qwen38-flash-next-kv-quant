#!/usr/bin/env python3
"""Numerischer Selbsttest fuer Stufe 2 (nvfp4) -- OHNE Serverstart.

Schwesterskript zu `kvq-kerneltest.py` (Stufe 1, fp8). Gleiche Begruendung:
ein vLLM-Start mit diesem Modell kostet 10-20 Minuten, und ein falsch
rechnender Dequant-Zweig zeigt sich danach nur als *Qualitaetszahl* -- also in
genau der Form, in der man ihn am leichtesten als "4 Bit kosten halt etwas"
fehldeutet. Der Test trennt die Fragen vorher.

Bei nvfp4 kommt eine Frage dazu, die es bei fp8 nicht gab: **wie liegen die
Bytes?** fp8 war ein Byte je Wert, fertig. nvfp4 hat

  * zwei Werte je Byte (E2M1-Nibbles, geradzahliger Index im niederwertigen),
  * eine E4M3-Blockskala je 16 Werte in einer GETRENNTEN Region derselben
    Seite (`[K_data | K_scale | V_data | V_scale]`),
  * und -- der unangenehme Teil -- eine **Verschraenkung** der Skalen, die der
    vorkompilierte CUDA-Schreiber nur fuer die V-Seite anwendet, der
    Triton-Software-Schreiber aus PR #44389 bei head_dim 256 dagegen fuer
    keine der beiden Seiten.

Ein Layout-Irrtum erzeugt keinen Absturz. Er erzeugt Zahlen. Deshalb wird das
Layout hier GEMESSEN und nicht angenommen:

  N0  Existiert der vorkompilierte nvfp4-Schreibpfad im Image ueberhaupt?
  N1  Layout-Sonde: ein Muster mit stark wechselnden Gruppen-Amplituden
      schreiben und unter ALLEN VIER Skalen-Hypothesen (K linear/verschraenkt
      x V linear/verschraenkt) zurueckrechnen. Auf der CPU vorab geprueft:
      die richtige Hypothese landet bei rel 0,090 (= reines
      NVFP4-Quantisierungsrauschen, deckt sich mit der ~9-%-Schaetzung des
      Plans), jede falsche bei rel 1,33 -- Faktor 15 Abstand.
  N2  Blockskalen-Statistik: Saettigung (>448) und Unterlauf (<2^-9). Das ist
      das nvfp4-Gegenstueck zur "0,0000 % ausserhalb +-448"-Zeile aus Stufe 1;
      bei 4 Bit traegt die Skala 1,0 nicht automatisch.
  N3  Lese-Kernel gegen float64-Referenz AUF DEN DEQUANTISIERTEN WERTEN --
      misst NUR den Kernel, die Quantisierung ist herausgerechnet.
  N4  Alle fuenf Kachel-Profile des Wrappers (2/5/64/200/300 Zeilen). Die
      Lehre aus Stufe 1: ein Test, der einen Zweig anfasst, belegt nur diesen
      Zweig -- der Prefill-Fall BLOCK_N=64 kippte dort an Shared Memory,
      nachdem der Decode-Fall laengst gruen war.
  N5  Software-Schreiber gegen den vorkompilierten Schreiber (Rueckfall S2b).
  N6  bf16-Pfad bitweise unveraendert.
  N7  Produktions-Seitengroesse: alles oben laeuft mit PAGE_SIZE 64, der
      Server faehrt aber 1600 (fp8, gemessen) bzw. voraussichtlich 2848
      (nvfp4) -- die Hybrid-Seitenausrichtung gegen die GDN-Seite hebt die
      Blockgroesse. Einmal in der echten Geometrie nachgerechnet.

Der PyTorch-Entpacker (`dequantisieren`) ist bewusst eine ZWEITE, unabhaengige
Implementierung -- waeren Entpacker und Kernel dieselbe Codezeile, wuerde der
Test nur beweisen, dass der Kernel mit sich selbst uebereinstimmt.

Aufruf im Container:  python3 /qfn/kvq-nvfp4-test.py
Rueckgabe 0 = bestanden. Die gemessene Layout-Hypothese wird am Ende als
fertige Umgebungszeile fuer den Serverstart ausgegeben.
"""

import os
import sys

import torch

# model zuerst -- qsa.py direkt zu importieren loest einen (vorbestehenden)
# Zirkelimport aus.
from vllm.models.qwen3_8_flash_next.nvidia import model as _model  # noqa: F401
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention
from vllm.utils.torch_utils import (
    nvfp4_kv_cache_full_dim,
    nvfp4_split_data_scale,
)

NUM_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
PAGE_SIZE = 64
TOPK = 2051
NUM_ROWS = 5
NUM_REQ = 2
SEQ_LEN = 4096
SEED = 0

# Trennschwelle der Layout-Sonde. Auf der CPU vorab gemessen: richtig 0,090 /
# falsch 1,33. 0,30 liegt mit grossem Abstand zwischen beiden.
LAYOUT_SCHWELLE = 0.30

# Produktions-Blockgroessen (N7). Gemessen: 800 bei bf16, 1600 bei fp8;
# 2848 ist die Vorhersage fuer nvfp4 (Hybrid-Seitenausrichtung gegen die
# GDN-Seite von 1.634.304 B). Ueberschreibbar per Umgebungsvariable.
PROD_PAGES = os.environ.get("QFN_KVQ_PROD_PAGES", "1600,2848")

# E2M1: Magnitude 0..7 -> 0, 0.5, 1, 1.5, 2, 3, 4, 6 (Bit 3 ist das Vorzeichen)
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_tabelle(device: torch.device) -> torch.Tensor:
    return torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.float32, device=device)


def sf_nach_float(bits: torch.Tensor) -> torch.Tensor:
    """E4M3-Bitmuster (Betrag) -> float32. Gegenstueck zu _qfn_sf_to_float."""
    p = (bits.to(torch.int32)) & 0x7F
    exp = (p >> 3) & 0x0F
    mant = p & 0x07
    normal = torch.pow(2.0, (exp.float() - 7.0)) * (1.0 + mant.float() / 8.0)
    subnormal = mant.float() / 512.0
    wert = torch.where(exp == 0, subnormal, normal)
    return torch.where(p == 0, torch.zeros_like(wert), wert)


def sf_koordinaten(page_size: int, scale_dim: int, swizzled: bool, device):
    """Ablage-Koordinaten (Zeile, Skalenindex) fuer jede logische (t, s).

    LINEAR: Identitaet. SWIZZLED (CUDA-Schreiber, V-Seite):
        (t, s) -> ((t//4)*4 + s//G, (s%G)*4 + t%4),  G = scale_dim//4
    """
    t = torch.arange(page_size, device=device).view(-1, 1)
    s = torch.arange(scale_dim, device=device).view(1, -1)
    if not swizzled:
        return t.expand(page_size, scale_dim), s.expand(page_size, scale_dim)
    g = scale_dim // 4
    return (t // 4) * 4 + (s // g), (s % g) * 4 + (t % 4)


def sichten(kv_cache: torch.Tensor, num_kv_heads: int):
    """(k_data, k_sf, v_data, v_sf) -- genau wie im Patch, uint8-Rohbits."""
    k_side = kv_cache[:, :num_kv_heads].transpose(1, 2)
    v_side = kv_cache[:, num_kv_heads:].transpose(1, 2)
    k_data, k_sf = nvfp4_split_data_scale(k_side)
    v_data, v_sf = nvfp4_split_data_scale(v_side)
    return k_data, k_sf.view(torch.uint8), v_data, v_sf.view(torch.uint8)


def dequantisieren(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    k_swizzled: bool,
    v_swizzled: bool,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    """Packt einen nvfp4-Cache in PyTorch aus. Rueckgabe: zwei bf16-Tensoren
    der Form [B, N, H, head_dim] -- also genau die Form, die der bf16-Kernel
    als K/V-Cache erwartet."""
    dev = kv_cache.device
    tabelle = e2m1_tabelle(dev)
    scale_dim = head_dim // 16
    page_size = kv_cache.shape[2]
    k_side = kv_cache[:, :num_kv_heads].transpose(1, 2)
    v_side = kv_cache[:, num_kv_heads:].transpose(1, 2)
    ergebnis = []
    for seite, swizzled, global_scale in (
        (k_side, k_swizzled, k_scale),
        (v_side, v_swizzled, v_scale),
    ):
        daten, skalen = nvfp4_split_data_scale(seite)
        daten = daten.contiguous()  # [B, N, H, head_dim//2]
        skalen = skalen.view(torch.uint8)
        b, n, h, _ = daten.shape
        niedrig = daten & 0x0F
        hoch = (daten >> 4) & 0x0F
        nibbles = torch.stack((niedrig, hoch), dim=-1).reshape(b, n, h, head_dim)
        werte = tabelle[nibbles.long()]
        t_idx, s_idx = sf_koordinaten(page_size, scale_dim, swizzled, dev)
        gelesen = skalen[:, t_idx.reshape(-1), :, :]
        gelesen = gelesen.reshape(b, page_size, scale_dim, h, scale_dim)
        s_sel = s_idx.view(1, page_size, scale_dim, 1, 1).expand(
            b, page_size, scale_dim, h, 1
        )
        gelesen = torch.gather(gelesen, 4, s_sel).squeeze(4)
        gelesen = gelesen.permute(0, 1, 3, 2)  # [B, N, H, SD]
        faktor = sf_nach_float(gelesen).repeat_interleave(16, dim=3)
        ergebnis.append((werte * faktor * global_scale).to(torch.bfloat16))
    return ergebnis[0], ergebnis[1]


def referenz(q, k_cache, v_cache, indices, block_table, token_to_req):
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
            keys = k_cache[p, o, kvh].to(torch.float64)
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
        f"  {name:<46} max {d.max().item():.3e}  "
        f"mittel {d.mean().item():.3e}  rel {(d.mean() / n).item():.3e}"
    )
    return (d.mean() / n).item()


def rel_fehler(ist: torch.Tensor, soll: torch.Tensor) -> float:
    return (
        (ist.float() - soll.float()).abs().mean() / soll.float().abs().mean()
    ).item()


def schreibe_nativ(kv_cache, key, value, num_kv_heads, slots, skala):
    from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

    kv_cache.zero_()
    reshape_and_cache_flash(
        key,
        value,
        kv_cache[:, :num_kv_heads].transpose(1, 2),
        kv_cache[:, num_kv_heads:].transpose(1, 2),
        slots,
        "nvfp4",
        skala,
        skala,
    )
    torch.cuda.synchronize()


def main() -> int:  # noqa: C901
    if not torch.cuda.is_available():
        print("FEHLER: keine CUDA-GPU sichtbar", file=sys.stderr)
        return 1
    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name(0)}  SM{cap[0]}{cap[1]}")
    voll = nvfp4_kv_cache_full_dim(HEAD_DIM)
    print(f"nvfp4_kv_cache_full_dim({HEAD_DIM}) = {voll}")
    if voll != HEAD_DIM // 2 + HEAD_DIM // 16:
        print("FEHLER: unerwartete Zellbreite", file=sys.stderr)
        return 1
    torch.manual_seed(SEED)

    pages_pro_req = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = NUM_REQ * pages_pro_req + 3
    num_slots = num_blocks * PAGE_SIZE
    fehler = 0

    q = (torch.randn(NUM_ROWS, NUM_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
    # Betriebsnahe K/V (Post-RMSNorm-Groessenordnung) fuer N2-N5.
    key = (torch.randn(num_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
    val = (torch.randn(num_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
    # Sonden-Muster fuer N1: jede 16er-Gruppe eine andere Groessenordnung, damit
    # eine falsch gelesene Blockskala um Faktoren danebenliegt, nicht knapp.
    t_i = torch.arange(num_slots, device=dev).view(-1, 1, 1)
    h_i = torch.arange(NUM_KV_HEADS, device=dev).view(1, -1, 1)
    g_i = torch.arange(HEAD_DIM // 16, device=dev).view(1, 1, -1)
    amp = torch.pow(4.0, ((t_i * 7 + h_i * 5 + g_i * 3) % 5).float()) * 0.05
    amp = amp.repeat_interleave(16, dim=2)
    key_sonde = (torch.randn(num_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * amp).bfloat16()
    val_sonde = (torch.randn(num_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * amp).bfloat16()

    slots = torch.arange(num_slots, dtype=torch.int64, device=dev)
    kv_cache = torch.zeros(
        num_blocks, 2 * NUM_KV_HEADS, PAGE_SIZE, voll, dtype=torch.uint8, device=dev
    )
    eins = torch.tensor(1.0, dtype=torch.float32, device=dev)

    # ------------------------------------------------------------------
    print("\n[N0] Vorkompilierter NVFP4-Schreibpfad im Image?")
    nativ_ok = False
    try:
        schreibe_nativ(kv_cache, key_sonde, val_sonde, NUM_KV_HEADS, slots, eins)
        nativ_ok = True
        print("  vorhanden und ausfuehrbar (reshape_and_cache_nvfp4_dispatch)")
    except Exception as e:  # noqa: BLE001
        print(f"  NICHT nutzbar: {type(e).__name__}: {e}")
        print("  -> Stufe 2 muss ueber den Software-Schreiber laufen")

    # ------------------------------------------------------------------
    k_swz, v_swz = False, True
    if nativ_ok:
        print("\n[N1] Layout-Sonde: welche Skalen-Verschraenkung schreibt das Image?")
        print("      K-Hypothese  V-Hypothese     rel. Fehler K   rel. Fehler V")
        beste = None
        for ks in (False, True):
            for vs in (False, True):
                kd, vd = dequantisieren(kv_cache, NUM_KV_HEADS, HEAD_DIM, ks, vs)
                fk = rel_fehler(kd.reshape(num_slots, NUM_KV_HEADS, HEAD_DIM), key_sonde)
                fv = rel_fehler(vd.reshape(num_slots, NUM_KV_HEADS, HEAD_DIM), val_sonde)
                if beste is None or fk + fv < beste[0]:
                    beste = (fk + fv, ks, vs, fk, fv)
                print(
                    f"      {'verschraenkt' if ks else 'linear':<12} "
                    f"{'verschraenkt' if vs else 'linear':<15} "
                    f"{fk:>13.4f}   {fv:>13.4f}"
                )
        _, k_swz, v_swz, fk, fv = beste
        print(
            f"  GEMESSEN: K {'verschraenkt' if k_swz else 'linear'}, "
            f"V {'verschraenkt' if v_swz else 'linear'} "
            f"(rel {fk:.4f} / {fv:.4f})"
        )
        if max(fk, fv) > LAYOUT_SCHWELLE:
            print(
                f"  FEHLGESCHLAGEN: beste Hypothese liegt bei {max(fk, fv):.3f} "
                f"(Schwelle {LAYOUT_SCHWELLE}) -- das Image schreibt ein Layout, "
                "das hier nicht abgebildet ist."
            )
            fehler += 1
        if k_swz:
            print(
                "  ACHTUNG: der Lese-Kernel liest K-Skalen fest LINEAR. "
                "Verschraenkte K-Skalen sind nicht implementiert."
            )
            fehler += 1
        # Cache jetzt mit den betriebsnahen Daten neu schreiben.
        schreibe_nativ(kv_cache, key, val, NUM_KV_HEADS, slots, eins)

    # ------------------------------------------------------------------
    print("\n[N2] Blockskalen-Statistik (traegt die Skala 1,0?)")
    gruppen_max = key.float().abs().reshape(num_slots, NUM_KV_HEADS, -1, 16).amax(-1)
    ziel = gruppen_max / 6.0
    ueber = (ziel > 448.0).float().mean().item()
    unter = ((ziel > 0) & (ziel < 2.0**-9)).float().mean().item()
    print(f"  Blockskala > 448 (E4M3-Saettigung):       {ueber * 100:.4f} %")
    print(f"  Blockskala < 2^-9 (E4M3-Unterlauf -> 0):  {unter * 100:.4f} %")
    print(f"  Bereich: {ziel.min().item():.3e} .. {ziel.max().item():.3e}")
    if ueber > 0 or unter > 0:
        print("  WARNUNG: Skala 1,0 traegt hier NICHT sauber -- Kalibrierung pruefen.")

    if not nativ_ok:
        print("\n[N3-N5] uebersprungen: ohne Schreiber gibt es keinen Cache-Inhalt.")
        return 3

    # ------------------------------------------------------------------
    print("\n[N3] Lese-Kernel gegen float64-Referenz auf den dequantisierten Werten")
    k_deq, v_deq = dequantisieren(kv_cache, NUM_KV_HEADS, HEAD_DIM, k_swz, v_swz)
    k_deq = k_deq.contiguous()
    v_deq = v_deq.contiguous()
    k_data, k_sf, v_data, v_sf = sichten(kv_cache, NUM_KV_HEADS)
    kb = key.reshape(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).contiguous()
    vb = val.reshape(num_blocks, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM).contiguous()

    block_table = torch.arange(
        NUM_REQ * pages_pro_req, dtype=torch.int32, device=dev
    ).view(NUM_REQ, pages_pro_req)
    token_to_req = torch.randint(0, NUM_REQ, (NUM_ROWS,), dtype=torch.int32, device=dev)
    indices = torch.randint(0, SEQ_LEN, (NUM_ROWS, TOPK), dtype=torch.int32, device=dev)
    indices[:, TOPK // 2 :] = -1

    ref_bf16 = referenz(q, kb, vb, indices, block_table, token_to_req)
    ref_nvfp4 = referenz(q, k_deq, v_deq, indices, block_table, token_to_req)
    out_bf16 = qsa_sparse_paged_attention(q, kb, vb, indices, block_table, token_to_req)
    out_nvfp4 = qsa_sparse_paged_attention(
        q,
        k_data,
        v_data,
        indices,
        block_table,
        token_to_req,
        k_scale=eins,
        v_scale=eins,
        k_scale_cache=k_sf,
        v_scale_cache=v_sf,
        v_sf_swizzled=v_swz,
    )
    print("\nVergleiche:")
    r_bf16 = bericht("Kernel bf16   gegen Referenz bf16", out_bf16, ref_bf16)
    r_kern = bericht("Kernel nvfp4  gegen Referenz nvfp4 (deq)", out_nvfp4, ref_nvfp4)
    r_quant = bericht("Kernel nvfp4  gegen Referenz bf16", out_nvfp4, ref_bf16)
    print(
        "\nDeutung: Zeile 2 misst NUR den Kernel (Quantisierung herausgerechnet)\n"
        "  und muss in der Groessenordnung von Zeile 1 liegen. Zeile 3 ist der\n"
        "  Preis von 4 Bit auf synthetischen N(0; 0,5)-Daten."
    )
    if r_kern > max(5 * r_bf16, 5e-3):
        print(f"\nFEHLGESCHLAGEN: Kernelfehler nvfp4 {r_kern:.3e} zu gross.")
        fehler += 1

    # ------------------------------------------------------------------
    print("\n[N4] Kachel-Profile des Wrappers (Ressourcen + Uebereinstimmung)")
    print("      Zeilen  Programme  BLOCK_N  Stufen   max|nvfp4 - bf16(deq)|")
    for rows in (2, 5, 64, 200, 300):
        programme = rows * NUM_KV_HEADS
        # QFN-PATCH kvq-s2b: die Anzeige muss den Uebersteuerungen folgen,
        # sonst meldet der Test 64/1, waehrend 32/2 laeuft. Sie bildet die
        # Wrapper-Regel nach -- die Wahrheit ueber den WIRKLICH kompilierten
        # Kernel steht in s2b-belegung.py, das den Triton-Cache ausliest.
        _bn_env = os.environ.get("QFN_KVQ_WIDE_BLOCK_N", "").strip()
        bn = 16 if programme < 32 else int(_bn_env or 64)
        _st_env = os.environ.get("QFN_KVQ_STAGES", "").strip()
        stufen = int(_st_env) if _st_env else (1 if bn > 16 else 2)
        qp = (torch.randn(rows, NUM_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
        ip = torch.randint(0, SEQ_LEN, (rows, TOPK), dtype=torch.int32, device=dev)
        ip[:, TOPK // 2 :] = -1
        tp = torch.randint(0, NUM_REQ, (rows,), dtype=torch.int32, device=dev)
        try:
            o_deq = qsa_sparse_paged_attention(qp, k_deq, v_deq, ip, block_table, tp)
            o_q = qsa_sparse_paged_attention(
                qp,
                k_data,
                v_data,
                ip,
                block_table,
                tp,
                k_scale=eins,
                v_scale=eins,
                k_scale_cache=k_sf,
                v_scale_cache=v_sf,
                v_sf_swizzled=v_swz,
            )
        except Exception as e:  # noqa: BLE001
            print(f"      {rows:>6}  {programme:>9}  {bn:>7}  {stufen:>6}   FEHLER: {e}")
            fehler += 1
            continue
        d = (o_q.to(torch.float64) - o_deq.to(torch.float64)).abs().max().item()
        marke = "ok" if d < 5e-3 else "ABWEICHUNG"
        print(f"      {rows:>6}  {programme:>9}  {bn:>7}  {stufen:>6}   {d:.3e}  {marke}")
        if d >= 5e-3:
            fehler += 1

    # ------------------------------------------------------------------
    print("\n[N5] Software-Schreiber gegen den vorkompilierten Schreiber")
    try:
        from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_nvfp4_store

        kv_soft = torch.zeros_like(kv_cache)
        sk_data, sk_sf, sv_data, sv_sf = sichten(kv_soft, NUM_KV_HEADS)
        qsa_nvfp4_store(
            key, val, sk_data, sk_sf, sv_data, sv_sf, slots, eins, eins,
            v_sf_swizzled=v_swz,
        )
        torch.cuda.synchronize()
        gleich = torch.equal(kv_soft, kv_cache)
        abweichend = (kv_soft != kv_cache).float().mean().item()
        print(
            f"  byteweise identisch: {gleich}  "
            f"(abweichende Bytes {abweichend * 100:.3f} %)"
        )
        sk_deq, sv_deq = dequantisieren(kv_soft, NUM_KV_HEADS, HEAD_DIM, k_swz, v_swz)
        rk = rel_fehler(sk_deq.reshape(num_slots, NUM_KV_HEADS, HEAD_DIM), key)
        rv = rel_fehler(sv_deq.reshape(num_slots, NUM_KV_HEADS, HEAD_DIM), val)
        print(f"  rel. Rekonstruktionsfehler Software-Schreiber: K {rk:.4f} / V {rv:.4f}")
        if max(rk, rv) > LAYOUT_SCHWELLE:
            print("  FEHLGESCHLAGEN: Software-Schreiber schreibt ein anderes Layout.")
            fehler += 1
    except Exception as e:  # noqa: BLE001
        print(f"  FEHLER: {type(e).__name__}: {e}")
        fehler += 1

    # ------------------------------------------------------------------
    print("\n[N6] bf16-Pfad bitweise unveraendert")
    a = qsa_sparse_paged_attention(q, kb, vb, indices, block_table, token_to_req)
    b = qsa_sparse_paged_attention(
        q, kb, vb, indices, block_table, token_to_req, k_scale=eins, v_scale=eins
    )
    bitgleich = torch.equal(a, b)
    print(f"  mit/ohne Skalen-Argumente identisch: {bitgleich}")
    if not bitgleich:
        fehler += 1

    # ------------------------------------------------------------------
    # N7 -- Produktions-Seitengroesse.
    #
    # Alles oben laeuft mit PAGE_SIZE 64. Die echte KV-Manager-Blockgroesse ist
    # das NICHT: der Hybrid-Allokator gleicht die Attention-Seite an die
    # GDN-Seite an und landet gemessen bei 800 (bf16) bzw. 1600 (fp8) Token; fuer
    # nvfp4 sind 2848 vorhergesagt. Die Verschraenkungsformel arbeitet zwar nur
    # mit t%4 und t//4 und ist damit blockgroessen-unabhaengig -- aber genau
    # solche "ist doch offensichtlich"-Stellen waren in Stufe 1 die teuren.
    # Also einmal in der Produktionsgeometrie nachgerechnet.
    print("\n[N7] Produktions-Seitengroesse (Blockgroesse wie im Server)")
    for prod_page in (int(x) for x in PROD_PAGES.split(",") if x.strip()):
        try:
            pb = 3
            ps_slots = pb * prod_page
            pk = (torch.randn(ps_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
            pv = (torch.randn(ps_slots, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.5).bfloat16()
            pslots = torch.arange(ps_slots, dtype=torch.int64, device=dev)
            pkv = torch.zeros(
                pb, 2 * NUM_KV_HEADS, prod_page, voll, dtype=torch.uint8, device=dev
            )
            schreibe_nativ(pkv, pk, pv, NUM_KV_HEADS, pslots, eins)
            pkd, pvd = dequantisieren(pkv, NUM_KV_HEADS, HEAD_DIM, k_swz, v_swz)
            rk = rel_fehler(pkd.reshape(ps_slots, NUM_KV_HEADS, HEAD_DIM), pk)
            rv = rel_fehler(pvd.reshape(ps_slots, NUM_KV_HEADS, HEAD_DIM), pv)
            marke = "ok" if max(rk, rv) <= LAYOUT_SCHWELLE else "ABWEICHUNG"
            print(f"      PAGE_SIZE {prod_page:>5}: rel K {rk:.4f} / V {rv:.4f}  {marke}")
            if max(rk, rv) > LAYOUT_SCHWELLE:
                fehler += 1
        except Exception as e:  # noqa: BLE001
            print(f"      PAGE_SIZE {prod_page:>5}: FEHLER {type(e).__name__}: {e}")
            fehler += 1

    print("\n" + "=" * 70)
    if fehler:
        print(f"FEHLGESCHLAGEN: {fehler} Punkt(e).")
        return 3
    print("BESTANDEN: nvfp4-Lesezweig korrekt, Layout gemessen, bf16 unveraendert.")
    print(f"  Quantisierungspreis nvfp4 (synthetisch): {r_quant:.3e}")
    print("Startzeile fuer den Server:")
    print(
        f'  QFN_KVQ_NVFP4_WRITER=native QFN_KVQ_V_SF_SWIZZLED={"1" if v_swz else "0"} '
        f"QFN_KV_DTYPE=nvfp4"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
