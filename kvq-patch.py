#!/usr/bin/env python3
"""Quant-KV-Patch fuer den QSA-Pfad von Qwen3.8-Flash-Next.

Stufe 1 (fp8_e4m3, ABGESCHLOSSEN + validiert 28.08.) und Stufe 2 (nvfp4) in
EINEM Skript. Welche Stufe aktiv wird, entscheidet allein `--kv-cache-dtype`:

    auto | bfloat16  -> unveraenderter bf16-Pfad (Zweige wegkompiliert)
    fp8 | fp8_e4m3   -> Stufe 1, 1,89x Kapazitaet
    nvfp4            -> Stufe 2, 3,10x Kapazitaet

Hintergrund, Messwerte und Betriebsregel: README.md in diesem Repository
(Stufe 1 = fp8_e4m3, Stufe 2 = nvfp4, Stufe 2b = Prefill-Kachel).

WAS HIER PASSIERT
-----------------
Qwen3.8-Flash-Next hat 12 QSA-Layer (block-sparse, 2 KV-Heads, head_dim 256).
Deren Haupt-KV wird von GENAU EINEM Kernel gelesen
(`_qsa_sparse_paged_gqa_splitk_kernel` in nvidia/ops/qsa.py) und von GENAU EINEM
Pfad geschrieben (geerbtes `do_kv_cache_update` -> `reshape_and_cache_flash`).
Der Schreibpfad kann fp8 UND nvfp4 laengst; die KV-Spec reicht `kv_quant_mode`
bereits durch; die Per-Layer-Skalen liegen als Device-Buffer am Layer
(`_k_scale`, `_v_scale`, Startwert 1,0). Es fehlt ausschliesslich (a) die
Dequantisierung im Lese-Kernel, (b) das Heben expliziter bf16-Verbote und
(c) fuer nvfp4 die Seiten-Geometrie (Spec) plus ein passender Schreib-Aufruf.

Der Indexer-Seitencache bleibt bf16 -- das ist KEINE Vorsicht, sondern
strukturell: `QSAKeyStateCache` packt exakte int64-mRoPE-Positionen als bf16-
Zellen in dieselben Zeilen (common/qsa_cache.py), eine Quantisierung wuerde die
Positionsbits zerstoeren. Ausserdem entscheidet der Indexer, welche Bloecke die
Attention ueberhaupt sieht -- Fehler dort sind Auswahlfehler, kein gradueller
Verlust. Deshalb ist der Kapazitaetsfaktor 1,89x bzw. 3,10x und nicht 2,0x/4,0x.

GEGEN DAS IMAGE GEBAUT, NICHT GEGEN DEN PR-BRANCH
-------------------------------------------------
Image `vllm/vllm-openai:qwen38-flash-next`, vLLM 0.1.dev20073+g8e685d198.
Dort heisst das Modul `vllm/models/qwen3_8_flash_next/` (im PR #53896:
`qwen4_exp`), die Klassen `Qwen3_8FlashNext*`. Jeder Anker unten ist am
Image-Dateisystem verifiziert; die Zaehl-Asserts brechen ab, sobald das Image
abweicht.

DIE DREI SPERREN, DIE DER PLAN NICHT KENNEN KONNTE (Stufe 1, alle geloest)
--------------------------------------------------------------------------
9.  `FlashAttentionImpl.__init__` (v1/attention/backends/flash_attn.py) wirft
    NotImplementedError, sobald `is_quantized_kv_cache(dtype)` gilt und
    `flash_attn_supports_kv_cache_dtype()` False liefert. Dieser Helfer ist
    "(FA3/FA4 auf SM90) oder (FA4 auf SM100)" -- SM120 kommt in keinem Zweig
    vor. Die Sperre ist fuer diesen Pfad sachlich unzustaendig: der QSA-Haupt-KV
    wird nie von einem FlashAttention-Kernel gelesen. Wir heben sie chirurgisch
    (dtype am Basis-Konstruktor vorbeireichen, danach zuruecksetzen).
10. vLLM legt JEDEN quantisierten KV als `torch.uint8` ab
    (STR_DTYPE_TO_TORCH_DTYPE). Die Bytes SIND e4m3 bzw. nvfp4, nur der
    Tensor-Typ traegt es nicht. Ohne Umdeutung laedt Triton Ganzzahlen 0..255
    und rechnet stillschweigend Unsinn.
11. Shared Memory: der fp8-Zweig verlangte mit BLOCK_N=64/num_stages=2 auf
    SM120 106.496 B bei 101.376 B Limit. Eine Pipeline-Stufe weniger, und zwar
    NUR im breiten Kachel-Fall, loest das (Decode behaelt zwei Stufen).

STUFE 2 -- WAS NVFP4 ZUSAETZLICH BRAUCHT
----------------------------------------
* **Seiten-Geometrie (S2a).** nvfp4 packt K und V in GETRENNTE Head-Slots und
  eine Zeile ist nur noch `head//2 + head//16` = 144 Byte gross (fp4-Daten +
  fp8-Block-Skalen, Blockgroesse 16) statt 2x256 bf16 = 1.024 Byte. Die Spec
  traegt dafuer die Felder `num_head_slots` (= 2 x num_kv_heads = 4) und
  `state_content_bytes` (= 144). Vorlage: `FlashInferBackend.customize_spec`.
  Wir setzen sie an ZWEI Stellen: als `customize_spec` am Backend (das ruft der
  gpu_model_runner) UND direkt im Owner-`get_kv_cache_spec` (der baut seine Spec
  selbst). Beide Wege sind durch `state_content_bytes is not None` gegen
  Doppelanwendung gesichert -- welcher Weg im Image tatsaechlich laeuft, ist
  damit egal.
* **Seiten-Layout.** Innerhalb einer Seite liegt
  `[K_data | K_scale | V_data | V_scale]`; die Daten ALLER Slots/Zeilen einer
  Seite stehen zusammenhaengend vor den Skalen (NICHT zeilenweise verschraenkt).
  `nvfp4_split_data_scale` (utils/torch_utils.py) liefert genau diese beiden
  Sichten strided und ohne Kopie -- wir rechnen also keine Byte-Offsets selbst.
* **Skalen-Verschraenkung (der kritische Punkt, Plan-Risiko R8).** Der
  vorkompilierte CUDA-Schreiber (`reshape_and_cache_nvfp4_kernel`) legt die
  K-Blockskalen LINEAR ab, die V-Blockskalen aber SWIZZLED
  ((t//4)*4 + s//4, (s%4)*4 + t%4) -- Letzteres fuer den SM100-trtllm-Kernel.
  Der Triton-Software-Schreiber aus PR #44389 legt bei head_dim 256 BEIDE
  linear ab. Beide Konventionen sind hier implementiert und ueber
  `QFN_KVQ_V_SF_SWIZZLED` umschaltbar; welche gilt, MISST der Selbsttest
  (`kvq-kerneltest.py`, Layout-Sonde) bevor ein Server startet.
* **Schreibpfad (S2b).** Der geerbte `do_kv_cache_update` teilt den Cache mit
  `.split(head_size, dim=-1)` -- das passt fuer nvfp4 NICHT (Zeile ist 144 B,
  K/V sitzen in verschiedenen Head-Slots). Deshalb ein Override im QSA-Impl,
  der die K/V-Seiten ueber die Head-Slots schneidet. Primaer laeuft danach der
  vorkompilierte CUDA-Schreiber; `QFN_KVQ_NVFP4_WRITER=triton` schaltet auf den
  arch-unabhaengigen Software-Schreiber um (Plan-Rueckfall S2b).
* **Lese-Kernel (S2c).** Zweiter constexpr-Zweig `KV_QUANT_NVFP4`: Byte-Loads
  (dim/2) + Skalen-Loads (dim/16), E2M1-Unpack und Skalen-Dekodierung nach
  PR #44389, dann bf16 -> Dots. NIE fp4/fp8 direkt in `tl.dot`.
* **Kein SM-Gate (S2d).** Der Software-Dequant ist arch-unabhaengig; anders als
  FlashInfer (family-100-only) braucht dieser Pfad keinen Capability-Guard.

Aufruf im Container (via qfn-vllm-start.sh, QFN_PATCHES="ple warmup kvq"):
    python3 /qfn/kvq-patch.py
Rueckgabe 0 = angewandt oder bereits gepatcht; !=0 = nichts veraendert.
"""

import os
import py_compile
import sys

MARKER = "# QFN-PATCH kvq-s1"

BASIS = "/usr/local/lib/python3.12/dist-packages/vllm"
PFAD_QSA = f"{BASIS}/models/qwen3_8_flash_next/nvidia/qsa.py"
PFAD_OPS = f"{BASIS}/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
PFAD_PLAT = f"{BASIS}/platforms/interface.py"
MARKER_PLAT = "# QFN-PATCH kvq-s2-plat"


# ---------------------------------------------------------------------------
# Datei 1 -- nvidia/qsa.py  (S1a Sperren + S1d Deklarationen + S2a/S2b)
# ---------------------------------------------------------------------------

QSA_EDITS = [
    # --- Import von is_quantized_kv_cache + Modul-Konstanten -----------------
    (
        "import+konstante",
        """from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer
""",
        """from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer

# QFN-PATCH kvq-s2: Nachtraegliche Importe. Nicht in die vorhandenen
# Import-Bloecke gemischt, damit jeder Anker klein und eindeutig bleibt.
import os as _qfn_os  # QFN-PATCH kvq-s2
from dataclasses import replace as _qfn_replace  # QFN-PATCH kvq-s2
from vllm.utils.torch_utils import (  # QFN-PATCH kvq-s2
    get_dtype_size as _qfn_dtype_size,
    nvfp4_kv_cache_full_dim as _qfn_nvfp4_full_dim,
    nvfp4_split_data_scale as _qfn_nvfp4_split,
)

# QFN-PATCH kvq-s1/s2: erlaubte Haupt-KV-dtypes. fp8_e5m2 bewusst NICHT -- fuer
# KV-Caches dem e4m3 unterlegen (kleinere Mantisse bei fuer K/V irrelevantem
# Exponentenumfang) und ohne Evidenzbedarf. nvfp4_4over6 ebenfalls nicht: die
# zweite FP4-Variante ist unbewertet und veraendert die Skalensuche im
# Schreiber, also genau die Groesse, die G3 messen soll. Der
# Indexer-Seitencache ist von dieser Liste nicht betroffen, er konstruiert
# seine Caches hart in bf16.
_QFN_KVQ_SUPPORTED = ("auto", "bfloat16", "fp8", "fp8_e4m3", "nvfp4")

# QFN-PATCH kvq-s1: erlaubte SPEICHER-dtypes des Haupt-KV. torch.uint8 ist hier
# kein Versehen: vLLM legt quantisierten KV grundsaetzlich als uint8 ab
# (STR_DTYPE_TO_TORCH_DTYPE["fp8_e4m3"] = torch.uint8, ebenso "nvfp4",
# utils/torch_utils.py). Die Bytes SIND e4m3 bzw. nvfp4, nur der Tensor-Typ
# traegt es nicht -- umgedeutet wird erst unmittelbar vor dem Kernel
# (forward_qsa), genau wie in MiniMax-M3.
_QFN_KVQ_STORAGE = (torch.bfloat16, torch.uint8, torch.float8_e4m3fn)

# QFN-PATCH kvq-s2: Schalter fuer die beiden nvfp4-Konventionen. Beide Werte
# werden vom Selbsttest gemessen, nicht geraten -- siehe kvq-kerneltest.py.
#   QFN_KVQ_NVFP4_WRITER  = "native" (vorkompilierter CUDA-Schreiber, Standard)
#                           "triton" (Software-Schreiber, arch-unabhaengig)
#   QFN_KVQ_V_SF_SWIZZLED = "1" (V-Blockskalen verschraenkt; so schreibt der
#                                CUDA-Schreiber) / "0" (linear; so schreibt der
#                                Triton-Schreiber bei head_dim 256)
_QFN_NVFP4_WRITER = _qfn_os.environ.get("QFN_KVQ_NVFP4_WRITER", "native").strip().lower()
_QFN_V_SF_SWIZZLED = _qfn_os.environ.get("QFN_KVQ_V_SF_SWIZZLED", "").strip()
if _QFN_V_SF_SWIZZLED == "":
    # Standard folgt dem Schreiber: nativ = verschraenkt, Triton = linear.
    _QFN_V_SF_SWIZZLED = _QFN_NVFP4_WRITER != "triton"
else:
    _QFN_V_SF_SWIZZLED = _QFN_V_SF_SWIZZLED not in ("0", "false", "no")


def _qfn_nvfp4_spec(spec):
    \"\"\"Seiten-Geometrie fuer nvfp4 nachziehen (Vorlage: FlashInferBackend).

    K und V bekommen je eigene Head-Slots (erst alle K-Koepfe, dann alle
    V-Koepfe -- dieselbe Reihenfolge, die FlashInfer mit
    ``kv_cache.split(num_kv_heads, dim=1)`` erwartet), und eine Zelle schrumpft
    von (head+head_v) bf16 auf ``head//2 + head//16`` Byte.

    Idempotent: ein bereits gesetztes ``state_content_bytes`` bleibt stehen --
    deshalb ist es egal, ob der Modell-Runner ``customize_spec`` zusaetzlich
    aufruft.
    \"\"\"
    if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
        return spec
    hs_k = _qfn_nvfp4_full_dim(spec.head_size)
    hs_v = _qfn_nvfp4_full_dim(spec.head_size_v)
    assert hs_k == hs_v, "nvfp4 with asymmetric K/V head sizes not yet supported"
    return _qfn_replace(
        spec,
        num_head_slots=2 * spec.num_kv_heads,
        state_content_bytes=hs_k * _qfn_dtype_size(spec.dtype),
    )


def _qfn_nvfp4_views(kv_cache: torch.Tensor, num_kv_heads: int):
    \"\"\"Zerlegt den gepackten nvfp4-Cache in vier Sichten (ohne Kopie).

    Eingang ist die logische Seite ``[B, H=2*num_kv_heads, N, 144]`` uint8.
    Ausgang: ``(k_data, k_sf, v_data, v_sf)``, je ``[B, N, num_kv_heads, X]``
    mit X = 128 (Daten) bzw. 16 (Blockskalen).

    Die Byte-Arithmetik macht `nvfp4_split_data_scale` aus vLLMs eigenem
    torch_utils -- bewusst NICHT selbst gerechnet, damit Schreiber (der
    dieselbe Zerlegung in C++ macht) und Leser per Konstruktion dasselbe
    Layout sehen. Die Skalen kommen als float8_e4m3fn zurueck; Triton soll die
    ROHEN BITS sehen (die Dekodierung passiert im Kernel), also zurueck auf
    uint8 umdeuten.
    \"\"\"
    k_side = kv_cache[:, :num_kv_heads].transpose(1, 2)
    v_side = kv_cache[:, num_kv_heads:].transpose(1, 2)
    k_data, k_sf = _qfn_nvfp4_split(k_side)
    v_data, v_sf = _qfn_nvfp4_split(v_side)
    return (
        k_data,
        k_sf.view(torch.uint8),
        v_data,
        v_sf.view(torch.uint8),
    )
""",
    ),
    # --- Sperre 1: Backend-Deklaration + eigener dtype-Check + customize_spec -
    (
        "backend-deklaration",
        """    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
""",
        """    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # QFN-PATCH kvq-s1/s2: fp8_e4m3 und nvfp4 fuer den Haupt-KV freigegeben.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "nvfp4",
    ]

    @classmethod
    def customize_spec(cls, spec):
        # QFN-PATCH kvq-s2: Der Modell-Runner reicht jede AttentionSpec durch
        # diese Methode (gpu_model_runner.get_kv_cache_spec) und die
        # Hybrid-Seitenausrichtung der Plattform benutzt sie ebenfalls, um die
        # Seitengroesse pro Token zu bestimmen. Ohne Override liefe hier die
        # Identitaet der Basisklasse -- die nvfp4-Seite waere dann viermal zu
        # gross deklariert und die GDN-Seitenpolsterung falsch.
        return _qfn_nvfp4_spec(spec)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # QFN-PATCH kvq-s2: Der v2-Modellrunner (v1/worker/gpu/attn_utils.py)
        # fragt die Cache-Form beim BACKEND ab und deutet den Rohpuffer damit
        # um -- er liest NICHT num_head_slots/state_content_bytes aus der Spec.
        # Ohne diesen Override versucht er die gepackten 144-Byte-Zeilen als
        # 2 Koepfe x 512 Byte zu lesen und stirbt mit
        #   RuntimeError: shape '[637, 2848, 2, 512]' is invalid for input of
        #   size 1044965376
        # -- wobei 1.044.965.376 = 637 x 4 x 2848 x 144 genau die gepackte
        # Groesse ist, die hier zurueckgegeben wird.
        if isinstance(cache_dtype_str, str) and cache_dtype_str.startswith("nvfp4"):
            return (
                num_blocks,
                2 * num_kv_heads,
                block_size,
                _qfn_nvfp4_full_dim(head_size),
            )
        return FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # QFN-PATCH kvq-s2: nvfp4 MUSS HND fahren, und das ist keine Vorliebe.
        # Das Seitenlayout ist [K_data | K_scale | V_data | V_scale]; die
        # K-Seite muss also eine zusammenhaengende halbe Seite sein. Unter NHD
        # (Vorgabe dieses Images) liegen K- und V-Head-Slots je Token
        # verschachtelt -- dann ist die K-Haelfte nicht mehr zusammenhaengend,
        # nvfp4_split_data_scale rechnet Schrittweiten fuer alle 2*num_kv_heads
        # Slots aus, und der Skalenbereich ueberlappt die Daten. Das gibt
        # keinen Absturz, sondern falsche Zahlen.
        # Fuer bf16/fp8 bleibt die geerbte Wahl unveraendert -- der
        # Stufe-1-Pfad ist davon per Konstruktion nicht betroffen.
        from vllm.config import get_current_vllm_config_or_none

        _cfg = get_current_vllm_config_or_none()
        _dtype = getattr(getattr(_cfg, "cache_config", None), "cache_dtype", "auto")
        if isinstance(_dtype, str) and _dtype.startswith("nvfp4"):
            # Identisch zur HND-Wahl von FlashAttentionBackend.
            if include_num_layers_dimension:
                return (1, 2, 0, 3, 4)
            return (0, 1, 2, 3)
        return FlashAttentionBackend.get_kv_cache_stride_order(
            include_num_layers_dimension
        )

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        # QFN-PATCH kvq-s1: NICHT die geerbte FlashAttention-Pruefung benutzen.
        # flash_attn_supports_kv_cache_dtype() ist "(FA3/FA4 auf SM90) oder
        # (FA4 auf SM100)" -- SM120 faellt durch. Fuer den QSA-Haupt-KV ist das
        # unzustaendig: gelesen wird ausschliesslich vom eigenen Triton-Kernel
        # _qsa_sparse_paged_gqa_splitk_kernel, der ab diesem Patch in Registern
        # dequantisiert; ein FlashAttention-Kernel sieht diesen Cache nie.
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(cls, *args, **kwargs) -> str | None:
        # QFN-PATCH kvq-s1: aus demselben Grund. Die geerbte Fassung meldet
        # "FP8 KV cache requires FA3 on SM90 or FA4 on SM100" -- eine Aussage
        # ueber FlashAttention-Kernel, die dieser Pfad nicht benutzt. Der Owner
        # verdrahtet sein Backend ohnehin hart, die Selektor-Pfade laufen fuer
        # qwen3_8_flash_next also gar nicht; der Override ist die Absicherung
        # fuer den Fall, dass doch ein Validierungspfad darueber stolpert.
        return None

    @staticmethod
    def get_name() -> str:
""",
    ),
    # --- Sperre 9 (neu): geerbter FA-Konstruktor -----------------------------
    (
        "impl-ctor-fa-sperre",
        """    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
""",
        """    def __init__(self, *args, **kwargs) -> None:
        # QFN-PATCH kvq-s1: FlashAttentionImpl.__init__ verwirft quantisierten
        # KV auf dieser Karte hart ("FlashAttention does not support fp8_e4m3
        # kv-cache on this device"), weil flash_attn_supports_kv_cache_dtype()
        # nur SM90 (FA3/FA4) und SM100 (FA4) kennt -- SM120 faellt durch. Die
        # Pruefung ist fuer QSA unzustaendig (eigener Triton-Leser, s. o.).
        # Wir halten den dtype am Basis-Konstruktor vorbei und setzen ihn danach
        # zurueck; gelesen wird er erst spaeter, im geerbten Schreibpfad
        # do_kv_cache_update -> reshape_and_cache_flash. kv_cache_dtype ist
        # Positional 6 in FlashAttentionImpl.__init__ (num_heads, head_size,
        # scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype).
        _qfn_kvq = None
        _qfn_roh = ("auto", "bfloat16")
        if len(args) > 6 and isinstance(args[6], str) and args[6] not in _qfn_roh:
            _qfn_kvq = args[6]
            args = tuple(args[:6]) + ("auto",) + tuple(args[7:])
        elif kwargs.get("kv_cache_dtype", "auto") not in _qfn_roh:
            _qfn_kvq = kwargs["kv_cache_dtype"]
            kwargs = dict(kwargs, kv_cache_dtype="auto")
        super().__init__(*args, **kwargs)
        if _qfn_kvq is not None:
            self.kv_cache_dtype = _qfn_kvq
        if not is_flash_attn_varlen_func_available():
""",
    ),
    # --- Sperre 2: Impl-Konstruktor -----------------------------------------
    (
        "impl-gate",
        """        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )
        self.supports_quant_query_input = False
""",
        """        # QFN-PATCH kvq-s1/s2
        if self.kv_cache_dtype not in _QFN_KVQ_SUPPORTED:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16, FP8-E4M3 or NVFP4 "
                "main KV cache"
            )
        self.use_fp8_kv = is_quantized_kv_cache(self.kv_cache_dtype)
        # QFN-PATCH kvq-s2: nvfp4 ist ein eigener Zweig, nicht "fp8 mit
        # anderem dtype" -- getrennte Seiten-Geometrie, getrennter Schreiber,
        # getrennter Lesezweig.
        self.qfn_nvfp4_kv = self.kv_cache_dtype.startswith("nvfp4")
        if self.qfn_nvfp4_kv:
            self.use_fp8_kv = False
        # QFN-PATCH kvq-s2: Sichten auf den gepackten Cache einmal je
        # Cache-Puffer bauen und merken. as_strided ist eine reine
        # Metadaten-Operation, aber sie laeuft sonst in JEDEM Decode-Schritt
        # neu; ausserdem bleiben die Zeiger so ueber CUDA-Graph-Capture hinweg
        # nachweislich dieselben.
        self._qfn_nvfp4_cache = None
        self.supports_quant_query_input = False
""",
    ),
    # --- S2b: Schreibpfad-Override (nvfp4) -----------------------------------
    (
        "do-kv-cache-update-override",
        """    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
""",
        """    def _qfn_nvfp4_cached_views(self, kv_cache: torch.Tensor):
        # QFN-PATCH kvq-s2: memoisierte Sichten, Schluessel ist der Zeiger auf
        # den Cache-Puffer (der wechselt nur bei Neuallokation).
        schluessel = kv_cache.data_ptr()
        gemerkt = self._qfn_nvfp4_cache
        if gemerkt is None or gemerkt[0] != schluessel:
            gemerkt = (schluessel, _qfn_nvfp4_views(kv_cache, self.num_kv_heads))
            self._qfn_nvfp4_cache = gemerkt
        return gemerkt[1]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # QFN-PATCH kvq-s2: Der geerbte Schreibpfad schneidet den Cache mit
        # `kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` -- das setzt
        # voraus, dass K und V in DERSELBEN Zeile liegen (Zeilenbreite
        # 2*head_size). Bei nvfp4 stimmt das nicht: K und V bekommen eigene
        # Head-Slots und die Zeile ist 144 Byte breit. Also hier schneiden,
        # nicht dort -- flash_attn.py bleibt unangetastet, alle anderen Modelle
        # im Image behalten ihr Verhalten.
        if not getattr(self, "qfn_nvfp4_kv", False):
            return super().do_kv_cache_update(
                layer, key, value, kv_cache, slot_mapping
            )
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return None
        n = self.num_kv_heads
        # (B, 2n, N, 144) -> zwei Sichten (B, N, n, 144), genau die Form, die
        # reshape_and_cache_nvfp4_dispatch erwartet: 4D, letzte Dimension
        # data_dim + scale_dim, Layout aus den Schrittweiten abgeleitet.
        key_cache = kv_cache[:, :n].transpose(1, 2)
        value_cache = kv_cache[:, n:].transpose(1, 2)
        if _QFN_NVFP4_WRITER == "triton":
            from .ops.qsa import qsa_nvfp4_store

            k_data, k_sf, v_data, v_sf = self._qfn_nvfp4_cached_views(kv_cache)
            qsa_nvfp4_store(
                key,
                value,
                k_data,
                k_sf,
                v_data,
                v_sf,
                slot_mapping,
                layer._k_scale,
                layer._v_scale,
                v_sf_swizzled=_QFN_V_SF_SWIZZLED,
            )
            return None
        # Lazy importiert, wie schon `from .ops.qsa import ...` weiter unten:
        # fa_utils definiert den Namen nur im CUDA-Zweig, und dieser Pfad soll
        # das Modul nicht schon beim Laden davon abhaengig machen.
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
        return None

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
""",
    ),
    # --- Sperre 3: Laufzeit-Check + Umdeutung in forward_qsa ------------------
    (
        "forward-dtype-check",
        """        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")
""",
        """        # QFN-PATCH kvq-s2: nvfp4 wird VOR dem normalen Schnitt behandelt --
        # die Zeile ist dort 144 Byte breit und traegt Daten UND Skalen, K und V
        # sitzen in getrennten Head-Slots. Es gibt also nichts bei head_size zu
        # teilen.
        _qfn_k_sf = None
        _qfn_v_sf = None
        if getattr(self, "qfn_nvfp4_kv", False):
            key_cache, _qfn_k_sf, value_cache, _qfn_v_sf = (
                self._qfn_nvfp4_cached_views(kv_cache)
            )
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        # QFN-PATCH kvq-s1: Der fp8-Cache liegt als uint8 vor (s. o.). Vor dem
        # Kernel auf den echten fp8-Typ umdeuten -- sonst laedt Triton die Bytes
        # als GANZZAHLEN 0..255 und rechnet stillschweigend Unsinn. view() ist
        # eine reine Typ-Umdeutung ohne Kopie (1 Byte auf 1 Byte, letzte Dimension
        # hat Schrittweite 1). Dasselbe Vorgehen wie MiniMax-M3
        # (common/sparse_attention.py: kv_cache.view(self.kv_cache_fp8_dtype)).
        # Bei nvfp4 bleibt es bei uint8: dort sind es gepackte Nibbles, kein
        # Gleitkommatyp, das Auspacken passiert im Kernel.
        if getattr(self, "use_fp8_kv", False):
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
        # Q bleibt bf16-Pflicht -- der Kernel rechnet seine Dots in bf16 und
        # quantisiert die Query bewusst nicht. K/V duerfen zusaetzlich fp8-e4m3
        # (Speicher: float8_e4m3fn) oder nvfp4 (Speicher: uint8) sein;
        # dequantisiert wird im Kernel vor den Dots.
        if query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 queries")
        if key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16, FP8-E4M3 or NVFP4 K/V"
            )
""",
    ),
    # --- Skalen an den Kernel durchreichen ----------------------------------
    (
        "kernel-aufruf-skalen",
        """        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
        )
""",
        """        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            # QFN-PATCH kvq-s1: Per-Layer-Skalen als Device-Buffer (Startwert
            # 1,0 aus set_default_quant_scales(register_buffer=True)). Device-
            # Buffer statt Host-Skalar ist der Grund, warum CUDA-Graph-Capture
            # hier unproblematisch ist -- kein host->device-Kopieren im Hot-Path.
            k_scale=getattr(layer, "_k_scale", None),
            v_scale=getattr(layer, "_v_scale", None),
            # QFN-PATCH kvq-s2: Blockskalen-Seiten (nur nvfp4, sonst None).
            k_scale_cache=_qfn_k_sf,
            v_scale_cache=_qfn_v_sf,
            v_sf_swizzled=_QFN_V_SF_SWIZZLED,
        )
""",
    ),
    # --- Sperre 4: Owner-__init__ (der Startblocker) -------------------------
    (
        "owner-gate",
        """        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )
""",
        """        # QFN-PATCH kvq-s1/s2
        if cache_config.cache_dtype not in _QFN_KVQ_SUPPORTED:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16, FP8-E4M3 or NVFP4 "
                "main KV cache"
            )
""",
    ),
    # --- Sperre 6: Cache-Storage-dtype --------------------------------------
    (
        "storage-gate",
        """        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"
            )
""",
        """        # QFN-PATCH kvq-s1
        if self.kv_cache_torch_dtype not in _QFN_KVQ_STORAGE:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16, FP8-E4M3 or NVFP4 "
                "cache storage"
            )
""",
    ),
    # --- S2a: Spec des Owners ------------------------------------------------
    (
        "owner-spec",
        """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
""",
        """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # QFN-PATCH kvq-s2: Der Owner baut seine Spec selbst -- also die
        # nvfp4-Geometrie hier gleich mit setzen und nicht darauf hoffen, dass
        # der Modell-Runner customize_spec aufruft. Beide Wege sind idempotent
        # (_qfn_nvfp4_spec laesst eine fertige Spec unveraendert), einer von
        # beiden greift auf jeden Fall.
        return _qfn_nvfp4_spec(
            FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                head_size_v=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            )
        )
""",
    ),
]


# ---------------------------------------------------------------------------
# Datei 2 -- nvidia/ops/qsa.py  (S1b Kernel-Dequant + S2b/S2c)
# ---------------------------------------------------------------------------

OPS_EDITS = [
    # --- nvfp4-Bausteine + Kernel-Signatur -----------------------------------
    (
        "kernel-signatur",
        """@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
""",
        '''# ---------------------------------------------------------------------------
# QFN-PATCH kvq-s2: NVFP4-Bausteine.
#
# Herkunft: vLLM PR #44389 ("Triton software NVFP4 KV cache", offen, 30.07.),
# Dateien v1/attention/ops/triton_unified_attention.py und
# triton_reshape_and_cache_flash.py. Uebernommen sind die reinen Bit-Helfer --
# sie sind arch-unabhaengig, brauchen KEINE nativen FP4-Instruktionen und
# genau deshalb auch keinen SM-Guard (Plan S2d). Die Index-Arithmetik ist
# NICHT uebernommen: sie muss zu unserem Seitenlayout passen, nicht zu deren.
# ---------------------------------------------------------------------------
import os as _qfn_ops_os  # QFN-PATCH kvq-s2b

# QFN-PATCH kvq-s2b -- HEBEL 1 aus S2 2.5 / 4.7: schmale Skalenkachel.
#
# Der S2c-Originalzweig laedt die Blockskalen als [HEAD_DIM, BLOCK_N], obwohl
# darin nur [HEAD_DIM/16, BLOCK_N] VERSCHIEDENE Werte stehen -- 16-facher
# Skalenverkehr, und diese Kachel ist genauso gross wie die gesamte
# fp8-Datenkachel. Hebel 1 laedt [SCALE_DIM, BLOCK_N] und weitet erst im
# Register auf; die E4M3-Dekodierung und die Multiplikation mit der
# Layer-Skala laufen dabei ebenfalls auf 16-mal weniger Elementen.
#
# Schalter, weil der Kontrollarm dieselbe Binaerdatei fahren muss:
#   QFN_KVQ_SF_NARROW = "1" (Vorgabe, Hebel 1 aktiv)
#                       "0" (S2c-Originalzweig, unveraendert -- Kontrollarm)
# Der Aufweitungs-Trick (broadcast_to + reshape) ist auf diesem Image gegen
# die naive breite Ladung BITGLEICH gemessen, ueber alle fuenf Kachel-Profile
# und fuer float32 wie uint8: kvq-logs/s2b-reshape-probe.log (Triton 3.7.1).
# Drei Modi, weil die Umordnung der Ladebefehle eine EIGENE Variable ist und
# nicht heimlich in Hebel 1 mitreisen darf:
#   QFN_KVQ_SF_MODE=0  S2c-Original, Wort fuer Wort: K vollstaendig (Bytes,
#                      Nibbles, Skalen, Rekonstruktion), danach V vollstaendig.
#   QFN_KVQ_SF_MODE=1  nur UMGEORDNET: beide Byte-Ladungen zuerst, danach die
#                      Skalenarbeit. Arithmetik identisch zu 0.
#   QFN_KVQ_SF_MODE=2  wie 1, zusaetzlich HEBEL 1 (schmale Skalenkachel).
#   QFN_KVQ_SF_MODE=3  wie 2, zusaetzlich HEBEL 2 (bytewise-Loader) auf
#                      BEIDEN Seiten (K mit tl.permute, V ohne).
#                      *** NICHT VALIDIERT -- LIEFERT BEI BLOCK_N=16 FALSCHE
#                      ZAHLEN. *** kvq-nvfp4-test.py N3/N4 schlagen fehl
#                      (Kernelfehler 1,586e+00 statt 2,085e-03; die beiden
#                      16er-Kachelprofile weichen um 1,0e-01 ab, die
#                      64er sind sauber). Der Zweig bleibt NUR zur weiteren
#                      Eingrenzung im Stand und darf NIE einen Server sehen.
#                      Eingegrenzt ist bereits: es liegt weder an der
#                      Wiederzusammensetzung (standalone bitgleich bis in
#                      tl.dot, alle Kachelbreiten) noch an den Pipelinestufen
#                      (Fehler identisch bei 1 und 2 Stufen).
#   QFN_KVQ_SF_MODE=4  wie 2, HEBEL 2 NUR auf der V-Seite (tl.join OHNE
#                      permute); K laedt klassisch doppelt.
#   QFN_KVQ_SF_MODE=5  wie 2, HEBEL 2 NUR auf der K-Seite (tl.join MIT
#                      tl.permute); V laedt klassisch doppelt.
#                      *** 4 und 5 sind DIAGNOSE-Zweige (S2 11.5, erster
#                      Spiegelstrich). Sie spalten den Fehler aus Modus 3 in
#                      seine beiden Traeger und duerfen ebenfalls NIE einen
#                      Server sehen -- unabhaengig davon, wie der Selbsttest
#                      ausgeht. Wer einen davon produktiv will, braucht eine
#                      eigene volle Gate-Kette.
# Vorgabe 2. Der Vergleich 0 gegen 1 misst die Umordnung, 1 gegen 2 den Hebel;
# 4 gegen 5 trennt, ob tl.permute der Traeger des Modus-3-Fehlers ist.
_QFN_SF_MODE = int(_qfn_ops_os.environ.get("QFN_KVQ_SF_MODE", "2").strip() or "2")


@triton.jit
def _qfn_decode_e2m1(nibble):
    """Ein FP4-E2M1-Nibble (Vorzeichen + 3 Magnitudenbits) nach float32.

    Kein Nachschlagen, sondern Bitbau: fuer Magnitude >= 2 ist der Wert
    2^(exp-1) * (1 + m/2), also (126 << 23) + (magnitude << 22) als
    float32-Bitmuster; Magnitude 0/1 sind die Subnormalen 0,0 und 0,5.
    """
    magnitude = nibble & 0x07
    magnitude_i32 = magnitude.to(tl.int32)
    sign_bits = ((nibble & 0x08).to(tl.uint32)) << 28
    normal_bits = ((126 << 23) + (magnitude_i32 << 22)).to(tl.uint32) | sign_bits
    normal = normal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    subnormal_bits = ((magnitude & 0x01).to(tl.uint32) * 0x3F000000) | sign_bits
    subnormal = subnormal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    return tl.where(magnitude < 2, subnormal, normal)


@triton.jit
def _qfn_sf_to_float(bits):
    """E4M3-Blockskala (nur Betrag) nach float32.

    Die Blockskalen entstehen aus Absolutmaxima, das Vorzeichenbit wird nicht
    benutzt; deshalb wird es maskiert statt ausgewertet. Bias 7 -> 127 ist die
    Addition von 120 auf den Exponenten; Subnormale sind mant * 2^-9.
    """
    payload = bits.to(tl.int32) & 0x7F
    exp_bits = (payload >> 3) & 0x0F
    mant = payload & 0x07
    normal_bits = ((exp_bits + 120) << 23) | (mant << 20)
    normal = normal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    subnormal = mant.to(tl.float32) / 512.0
    value = tl.where(exp_bits == 0, subnormal, normal)
    return tl.where(payload == 0, 0.0, value)


@triton.jit
def _qfn_float_to_sf_bits(x):
    """float32 -> E4M3-Bitmuster (nicht-negativ), Rundung zur naechsten Zahl."""
    x = tl.clamp(x, 0.0, 448.0)
    x_safe = tl.maximum(x, 0.0000000001)
    subnormal_mant = tl.floor(x * 512.0 + 0.5).to(tl.int32)
    subnormal_mant = tl.minimum(subnormal_mant, 7)
    x_bits = x_safe.to(tl.uint32, bitcast=True)
    exp_unbiased = ((x_bits >> 23) & 0xFF).to(tl.int32) - 127
    exp_bits = exp_unbiased + 7
    mant = (((x_bits & 0x7FFFFF) + 0x80000) >> 20).to(tl.int32)
    exp_bits += mant >> 3
    mant = mant & 7
    normal_bits = (exp_bits << 3) | mant
    bits = tl.where(x < 0.015625, subnormal_mant, normal_bits)
    bits = tl.where(x == 0.0, 0, bits)
    return bits.to(tl.uint8)


@triton.jit
def _qfn_float_to_e2m1_bits(x):
    """float32 -> FP4-E2M1-Nibble; Grenzen wie cvt.rn.satfinite.e2m1x2.f32."""
    sign = tl.where(x < 0.0, 8, 0)
    abs_x = tl.abs(x)
    mag = tl.full(x.shape, 0, dtype=tl.int32)
    mag = tl.where((abs_x > 0.25) & (abs_x < 0.75), 1, mag)
    mag = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 2, mag)
    mag = tl.where((abs_x > 1.25) & (abs_x < 1.75), 3, mag)
    mag = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 4, mag)
    mag = tl.where((abs_x > 2.5) & (abs_x < 3.5), 5, mag)
    mag = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 6, mag)
    mag = tl.where(abs_x > 5.0, 7, mag)
    return (sign | mag).to(tl.uint8)


@triton.jit
def _qfn_sf_coord(slot, group, SWIZZLED: tl.constexpr, SCALE_DIM: tl.constexpr):
    """Wo liegt die Blockskala fuer (Zeile im Block, 16er-Gruppe)?

    LINEAR ist die naheliegende Ablage (Zeile, Gruppe). SWIZZLED ist die
    Verschraenkung, die der vorkompilierte CUDA-Schreiber fuer die V-Seite
    benutzt (fuer den SM100-trtllm-Kernel):
        (t, s) -> ((t//4)*4 + s//G, (s%G)*4 + t%4),  G = SCALE_DIM//4
    Das ist eine Permutation innerhalb je vier Zeilen; welche Konvention im
    Image tatsaechlich gilt, MISST kvq-kerneltest.py, geraten wird nichts.
    """
    SWIZZLE_GROUP: tl.constexpr = SCALE_DIM // 4
    if SWIZZLED:
        return (slot // 4) * 4 + (group // SWIZZLE_GROUP), (
            group % SWIZZLE_GROUP
        ) * 4 + (slot % 4)
    return slot + group * 0, group + slot * 0


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    # QFN-PATCH kvq-s1: 1-Element-fp32-Device-Buffer je Layer (Skalar-Fall).
    # Per-Head-Skalen waeren ein spaeterer Ausbau; der Checkpoint liefert
    # ohnehin keine kalibrierten KV-Skalen, wir starten bei 1,0.
    k_scale_ptr,
    v_scale_ptr,
    # QFN-PATCH kvq-s2: Blockskalen-Seiten (uint8-Bits, nur im nvfp4-Zweig
    # gelesen; sonst wird ein beliebiger gueltiger Zeiger uebergeben).
    k_sf_ptr,
    v_sf_ptr,
    indices_ptr,
''',
    ),
    # --- Skalen-Schrittweiten in der Signatur --------------------------------
    (
        "kernel-sf-strides",
        """    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
""",
        """    stride_v_block,
    stride_v_token,
    stride_v_head,
    # QFN-PATCH kvq-s2: Schrittweiten der Blockskalen-Seiten. Sie stammen aus
    # nvfp4_split_data_scale und sind NICHT aus den Datenschrittweiten
    # ableitbar (Daten- und Skalenbereich haben verschiedene Zeilenbreiten,
    # 128 gegen 16 Byte).
    stride_ks_block,
    stride_ks_token,
    stride_ks_head,
    stride_vs_block,
    stride_vs_token,
    stride_vs_head,
    stride_indices_row,
""",
    ),
    # --- Kernel-Signatur: constexpr-Schalter ---------------------------------
    (
        "kernel-constexpr",
        """    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
""",
        """    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # QFN-PATCH kvq-s1: Compile-Zeit-Schalter. False kompiliert den Dequant-
    # Zweig restlos weg -- der bf16-Pfad bleibt damit bitweise unveraendert.
    USE_FP8: tl.constexpr = False,
    # QFN-PATCH kvq-s2: zweiter, voellig getrennter Lesezweig (andere Loads,
    # andere Breiten). Genau einer von USE_FP8 / KV_QUANT_NVFP4 ist gesetzt.
    KV_QUANT_NVFP4: tl.constexpr = False,
    V_SF_SWIZZLED: tl.constexpr = True,
    # QFN-PATCH kvq-s2b: 0 = S2c-Original, 1 = nur umgeordnet, 2 = zusaetzlich
    # Hebel 1 (schmale Skalenkachel), 3/4/5 = Hebel-2-Diagnosezweige (beide
    # Seiten / nur V / nur K). Alle stehen im selben Binaerstand, damit jeder
    # Kontrollarm NICHTS ausser diesem einen constexpr aendert.
    SF_MODE: tl.constexpr = 2,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
""",
    ),
    # --- Lade- und Dequant-Zweig ---------------------------------------------
    (
        "kernel-dequant",
        """        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
""",
        """        if KV_QUANT_NVFP4:
            # QFN-PATCH kvq-s2: NVFP4-Zweig.
            #
            # Eine Zeile traegt HEAD_DIM Werte in HEAD_DIM/2 Byte (zwei
            # E2M1-Nibbles je Byte, geradzahliger Index im NIEDERWERTIGEN)
            # plus HEAD_DIM/16 E4M3-Blockskalen in einer GETRENNTEN Seite.
            # Rekonstruktion: nibble * blockskala * layer-skala.
            #
            # Die Orientierungen bleiben wie im bf16-Zweig: keys als
            # [HEAD_DIM, BLOCK_N] (K geht transponiert in tl.dot), values als
            # [BLOCK_N, HEAD_DIM].
            #
            # HEAD_DIM // 16 wird als constexpr-Ausdruck direkt uebergeben --
            # eine `SCALE_DIM: tl.constexpr = ...`-Bindung INNERHALB eines
            # constexpr-Zweiges ist in Triton nicht zuverlaessig.
            byte_offsets = dim_offsets // 2
            if SF_MODE == 0:
                # QFN-PATCH kvq-s2b: MODUS 0 -- der S2c-Zweig vom 28.08.,
                # Anweisung fuer Anweisung in der urspruenglichen Reihenfolge
                # (K komplett, dann V komplett). Er bleibt im Binaerstand, weil
                # sonst nicht entscheidbar waere, ob ein gemessener Gewinn vom
                # Hebel kommt oder von der Umordnung der Ladebefehle.
                group_offsets = dim_offsets // 16
                k_bytes = tl.load(
                    k_cache_ptr
                    + safe_page[None, :] * stride_k_block
                    + page_offset[None, :] * stride_k_token
                    + kv_head * stride_k_head
                    + byte_offsets[:, None],
                    mask=valid[None, :],
                    other=0,
                )
                k_nib = tl.where(
                    (dim_offsets[:, None] & 1) == 0,
                    k_bytes & 0x0F,
                    (k_bytes >> 4) & 0x0F,
                )
                ks_slot, ks_group = _qfn_sf_coord(
                    page_offset[None, :], group_offsets[:, None], False, HEAD_DIM // 16
                )
                k_sf = tl.load(
                    k_sf_ptr
                    + safe_page[None, :] * stride_ks_block
                    + ks_slot * stride_ks_token
                    + kv_head * stride_ks_head
                    + ks_group,
                    mask=valid[None, :],
                    other=0,
                )
                keys = (
                    _qfn_decode_e2m1(k_nib)
                    * _qfn_sf_to_float(k_sf)
                    * tl.load(k_scale_ptr)
                ).to(query.dtype)
                v_bytes = tl.load(
                    v_cache_ptr
                    + safe_page[:, None] * stride_v_block
                    + page_offset[:, None] * stride_v_token
                    + kv_head * stride_v_head
                    + byte_offsets[None, :],
                    mask=valid[:, None],
                    other=0,
                )
                v_nib = tl.where(
                    (dim_offsets[None, :] & 1) == 0,
                    v_bytes & 0x0F,
                    (v_bytes >> 4) & 0x0F,
                )
                vs_slot, vs_group = _qfn_sf_coord(
                    page_offset[:, None], group_offsets[None, :], V_SF_SWIZZLED,
                    HEAD_DIM // 16,
                )
                v_sf = tl.load(
                    v_sf_ptr
                    + safe_page[:, None] * stride_vs_block
                    + vs_slot * stride_vs_token
                    + kv_head * stride_vs_head
                    + vs_group,
                    mask=valid[:, None],
                    other=0,
                )
                values = (
                    _qfn_decode_e2m1(v_nib)
                    * _qfn_sf_to_float(v_sf)
                    * tl.load(v_scale_ptr)
                ).to(query.dtype)
            else:
                if SF_MODE == 3:
                    # QFN-PATCH kvq-s2b -- HEBEL 2 (bytewise-Loader,
                    # Vorbild PR #44389). Der Zweig oben adressiert jedes
                    # Datenbyte ZWEIMAL: die Kachel laeuft ueber dim_offsets,
                    # und byte_offsets = dim_offsets // 2 traegt fuer die
                    # Zeilen 2i und 2i+1 dieselbe Adresse. Hier wird die halbe
                    # Kachel EINMAL geladen und die volle aus den beiden
                    # Nibbles zusammengesetzt.
                    #
                    # Das Zusammensetzen ist auf beiden Seiten VERSCHIEDEN und
                    # ist gemessen, nicht angenommen (s2b-bytewise-probe.log,
                    # bitgleich ueber alle fuenf Kachel-Profile):
                    #   K: [H/2, BLOCK_N] -> tl.join gibt [H/2, BLOCK_N, 2],
                    #      die 2 sitzt hinten -> permute (0,2,1) -> reshape.
                    #   V: [BLOCK_N, H/2] -> tl.join gibt [BLOCK_N, H/2, 2],
                    #      das ist bereits die gesuchte Ordnung, kein permute.
                    halb_offsets = tl.arange(0, HEAD_DIM // 2)
                    k_roh = tl.load(
                        k_cache_ptr
                        + safe_page[None, :] * stride_k_block
                        + page_offset[None, :] * stride_k_token
                        + kv_head * stride_k_head
                        + halb_offsets[:, None],
                        mask=valid[None, :],
                        other=0,
                    )
                    k_nib = tl.reshape(
                        tl.permute(
                            tl.join(k_roh & 0x0F, (k_roh >> 4) & 0x0F), (0, 2, 1)
                        ),
                        (HEAD_DIM, BLOCK_N),
                    )
                    v_roh = tl.load(
                        v_cache_ptr
                        + safe_page[:, None] * stride_v_block
                        + page_offset[:, None] * stride_v_token
                        + kv_head * stride_v_head
                        + halb_offsets[None, :],
                        mask=valid[:, None],
                        other=0,
                    )
                    v_nib = tl.reshape(
                        tl.join(v_roh & 0x0F, (v_roh >> 4) & 0x0F),
                        (BLOCK_N, HEAD_DIM),
                    )
                elif SF_MODE == 4:
                    # QFN-PATCH kvq-s2b -- HEBEL-2-DIAGNOSE, NUR V-Seite.
                    #
                    # S2 11.5, erster Spiegelstrich: Modus 3 rechnet bei
                    # BLOCK_N=16 falsch. Modus 3 fasst ZWEI Konstruktionen
                    # zugleich an -- K braucht tl.permute, V nicht. Dieser
                    # Zweig nimmt nur die V-Seite (ohne permute) und laesst K
                    # exakt so, wie Modus 2 ihn faehrt. Zusammen mit Modus 5
                    # trennt das die beiden Traeger in EINEM Selbsttestlauf.
                    halb_offsets = tl.arange(0, HEAD_DIM // 2)
                    k_bytes = tl.load(
                        k_cache_ptr
                        + safe_page[None, :] * stride_k_block
                        + page_offset[None, :] * stride_k_token
                        + kv_head * stride_k_head
                        + byte_offsets[:, None],
                        mask=valid[None, :],
                        other=0,
                    )
                    k_nib = tl.where(
                        (dim_offsets[:, None] & 1) == 0, k_bytes & 0x0F, (k_bytes >> 4) & 0x0F
                    )
                    v_roh = tl.load(
                        v_cache_ptr
                        + safe_page[:, None] * stride_v_block
                        + page_offset[:, None] * stride_v_token
                        + kv_head * stride_v_head
                        + halb_offsets[None, :],
                        mask=valid[:, None],
                        other=0,
                    )
                    v_nib = tl.reshape(
                        tl.join(v_roh & 0x0F, (v_roh >> 4) & 0x0F),
                        (BLOCK_N, HEAD_DIM),
                    )
                elif SF_MODE == 5:
                    # QFN-PATCH kvq-s2b -- HEBEL-2-DIAGNOSE, NUR K-Seite.
                    # Gegenstueck zu Modus 4: hier steht tl.permute drin, die
                    # V-Seite bleibt klassisch. Faellt 5 und 4 nicht, ist
                    # tl.permute der Traeger; faellt 4 und 5 nicht, ist es die
                    # V-Konstruktion; fallen beide, traegt tl.join selbst.
                    halb_offsets = tl.arange(0, HEAD_DIM // 2)
                    k_roh = tl.load(
                        k_cache_ptr
                        + safe_page[None, :] * stride_k_block
                        + page_offset[None, :] * stride_k_token
                        + kv_head * stride_k_head
                        + halb_offsets[:, None],
                        mask=valid[None, :],
                        other=0,
                    )
                    k_nib = tl.reshape(
                        tl.permute(
                            tl.join(k_roh & 0x0F, (k_roh >> 4) & 0x0F), (0, 2, 1)
                        ),
                        (HEAD_DIM, BLOCK_N),
                    )
                    v_bytes = tl.load(
                        v_cache_ptr
                        + safe_page[:, None] * stride_v_block
                        + page_offset[:, None] * stride_v_token
                        + kv_head * stride_v_head
                        + byte_offsets[None, :],
                        mask=valid[:, None],
                        other=0,
                    )
                    v_nib = tl.where(
                        (dim_offsets[None, :] & 1) == 0, v_bytes & 0x0F, (v_bytes >> 4) & 0x0F
                    )
                else:
                    k_bytes = tl.load(
                        k_cache_ptr
                        + safe_page[None, :] * stride_k_block
                        + page_offset[None, :] * stride_k_token
                        + kv_head * stride_k_head
                        + byte_offsets[:, None],
                        mask=valid[None, :],
                        other=0,
                    )
                    k_nib = tl.where(
                        (dim_offsets[:, None] & 1) == 0, k_bytes & 0x0F, (k_bytes >> 4) & 0x0F
                    )
                    v_bytes = tl.load(
                        v_cache_ptr
                        + safe_page[:, None] * stride_v_block
                        + page_offset[:, None] * stride_v_token
                        + kv_head * stride_v_head
                        + byte_offsets[None, :],
                        mask=valid[:, None],
                        other=0,
                    )
                    v_nib = tl.where(
                        (dim_offsets[None, :] & 1) == 0, v_bytes & 0x0F, (v_bytes >> 4) & 0x0F
                    )
                if SF_MODE >= 2:
                    # QFN-PATCH kvq-s2b -- HEBEL 1.
                    #
                    # Die Skalenkachel wird in ihrer NATUERLICHEN Breite geladen,
                    # [SCALE_DIM, BLOCK_N] statt [HEAD_DIM, BLOCK_N]. Dass das
                    # zulaessig ist, folgt aus zwei Dingen und nicht aus einem
                    # Gefuehl: dim_offsets ist tl.arange(0, HEAD_DIM) (Image-Quelle
                    # ops/qsa.py:236), und die Gruppe ist dim // 16 -- also 16
                    # AUFEINANDERFOLGENDE Zeilen je Gruppe. Genau diese Ordnung
                    # stellt broadcast_to + reshape wieder her.
                    #
                    # Zusatzgewinn, der nicht im Ladeverkehr steckt: _qfn_sf_to_float
                    # (rund acht Bit-Operationen) und die Multiplikation mit der
                    # Layer-Skala laufen jetzt auf SCALE_DIM statt HEAD_DIM Zeilen.
                    #
                    # Die Reihenfolge der Multiplikationen aendert sich damit von
                    # (nib * sf) * layer zu nib * (sf * layer). float32-Multiplikation
                    # ist nicht assoziativ -- der Unterschied liegt im letzten Bit
                    # und wird von N3/N7 gemessen, nicht behauptet.
                    sf_groups = tl.arange(0, HEAD_DIM // 16)
                    ks_slot, ks_group = _qfn_sf_coord(
                        page_offset[None, :], sf_groups[:, None], False, HEAD_DIM // 16
                    )
                    k_sf_f = _qfn_sf_to_float(
                        tl.load(
                            k_sf_ptr
                            + safe_page[None, :] * stride_ks_block
                            + ks_slot * stride_ks_token
                            + kv_head * stride_ks_head
                            + ks_group,
                            mask=valid[None, :],
                            other=0,
                        )
                    ) * tl.load(k_scale_ptr)
                    keys = (
                        _qfn_decode_e2m1(k_nib)
                        * tl.reshape(
                            tl.broadcast_to(
                                k_sf_f[:, None, :], (HEAD_DIM // 16, 16, BLOCK_N)
                            ),
                            (HEAD_DIM, BLOCK_N),
                        )
                    ).to(query.dtype)
                    vs_slot, vs_group = _qfn_sf_coord(
                        page_offset[:, None], sf_groups[None, :], V_SF_SWIZZLED,
                        HEAD_DIM // 16,
                    )
                    v_sf_f = _qfn_sf_to_float(
                        tl.load(
                            v_sf_ptr
                            + safe_page[:, None] * stride_vs_block
                            + vs_slot * stride_vs_token
                            + kv_head * stride_vs_head
                            + vs_group,
                            mask=valid[:, None],
                            other=0,
                        )
                    ) * tl.load(v_scale_ptr)
                    values = (
                        _qfn_decode_e2m1(v_nib)
                        * tl.reshape(
                            tl.broadcast_to(
                                v_sf_f[:, :, None], (BLOCK_N, HEAD_DIM // 16, 16)
                            ),
                            (BLOCK_N, HEAD_DIM),
                        )
                    ).to(query.dtype)
                else:
                    # QFN-PATCH kvq-s2b: der S2c-Originalzweig, unveraendert. Er
                    # bleibt im Binaerstand, damit der Kontrollarm dieser
                    # Optimierung KEINE zweite Variable traegt.
                    group_offsets = dim_offsets // 16
                    # K-Blockskalen liegen linear (so schreibt sie der
                    # CUDA-Schreiber, und so auch der Software-Schreiber).
                    ks_slot, ks_group = _qfn_sf_coord(
                        page_offset[None, :], group_offsets[:, None], False, HEAD_DIM // 16
                    )
                    k_sf = tl.load(
                        k_sf_ptr
                        + safe_page[None, :] * stride_ks_block
                        + ks_slot * stride_ks_token
                        + kv_head * stride_ks_head
                        + ks_group,
                        mask=valid[None, :],
                        other=0,
                    )
                    keys = (
                        _qfn_decode_e2m1(k_nib)
                        * _qfn_sf_to_float(k_sf)
                        * tl.load(k_scale_ptr)
                    ).to(query.dtype)
                    vs_slot, vs_group = _qfn_sf_coord(
                        page_offset[:, None], group_offsets[None, :], V_SF_SWIZZLED,
                        HEAD_DIM // 16,
                    )
                    v_sf = tl.load(
                        v_sf_ptr
                        + safe_page[:, None] * stride_vs_block
                        + vs_slot * stride_vs_token
                        + kv_head * stride_vs_head
                        + vs_group,
                        mask=valid[:, None],
                        other=0,
                    )
                    values = (
                        _qfn_decode_e2m1(v_nib)
                        * _qfn_sf_to_float(v_sf)
                        * tl.load(v_scale_ptr)
                    ).to(query.dtype)
        else:
            keys = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None],
                mask=valid[None, :],
                other=0.0,
            )
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if USE_FP8:
                # QFN-PATCH kvq-s1 -- wortgleich zu
                # minimax_m3/common/ops/sparse_attn.py. NIE fp8 direkt in
                # tl.dot: Triton kann fp8e4nv auf SM12x nicht multiplizieren
                # (derselbe Befund, der SGLang #36545 ausgeloest hat). Erst auf
                # den Query-dtype heben, dann mit der Layer-Skala multiplizieren
                # und zurueckcasten. Der Umweg kostet nur Register, gegen die
                # halbierte Ladebandbreite ist das nichts.
                keys = keys.to(query.dtype)
                keys = (keys * tl.load(k_scale_ptr)).to(query.dtype)
                values = values.to(query.dtype)
                values = (values * tl.load(v_scale_ptr)).to(query.dtype)
        scores = tl.dot(query, keys)
""",
    ),
    # --- Wrapper-Signatur ----------------------------------------------------
    (
        "wrapper-signatur",
        """    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    \"\"\"Run sparse GQA directly over paged BF16 K/V caches.\"\"\"
""",
        """    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    # QFN-PATCH kvq-s1: Device-Skalen des Layers. None = reiner bf16-Pfad.
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    # QFN-PATCH kvq-s2: Blockskalen-Seiten. Gesetzt = nvfp4-Pfad; dann sind
    # k_cache/v_cache die DATENSEITEN (uint8, letzte Dimension head_dim/2).
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
    v_sf_swizzled: bool = True,
) -> torch.Tensor:
    \"\"\"Run sparse GQA over paged BF16, FP8-E4M3 or NVFP4 K/V caches.\"\"\"
""",
    ),
    # --- Wrapper-Formpruefung (head_dim gegen Zeilenbreite) ------------------
    (
        "wrapper-formcheck",
        """    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
""",
        """    # QFN-PATCH kvq-s2: Im nvfp4-Fall traegt eine Cache-Zeile head_dim/2 Byte
    # (zwei Werte je Byte); die Blockskalen stehen in einer eigenen Seite.
    _qfn_nvfp4 = k_scale_cache is not None
    _qfn_zeilenbreite = q.shape[2] // 2 if _qfn_nvfp4 else q.shape[2]
    if _qfn_zeilenbreite != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
""",
    ),
    # --- Wrapper-Assert ------------------------------------------------------
    (
        "wrapper-assert",
        """    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
""",
        """    # QFN-PATCH kvq-s1/s2: Q bleibt bf16, K/V duerfen fp8-e4m3 (float8_e4m3fn)
    # oder nvfp4 (uint8, gepackt) sein.
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    assert k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)
    _qfn_use_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if _qfn_nvfp4:
        if k_cache.dtype != torch.uint8:
            raise ValueError("QSA nvfp4 KV cache must be stored as uint8")
        if v_scale_cache is None:
            raise ValueError("QSA nvfp4 KV cache requires both scale pages")
        if k_scale_cache.dtype != torch.uint8 or v_scale_cache.dtype != torch.uint8:
            raise ValueError("QSA nvfp4 block scales must be raw uint8 bits")
        if k_scale_cache.shape[3] != head_dim // 16:
            raise ValueError("QSA nvfp4 block-scale page has the wrong width")
        if k_scale_cache.shape[:3] != k_cache.shape[:3]:
            raise ValueError("QSA nvfp4 data and scale pages disagree in shape")
        if k_scale_cache.stride(3) != 1 or v_scale_cache.stride(3) != 1:
            raise ValueError("QSA nvfp4 block scales must be contiguous per row")
    elif k_cache.dtype == torch.uint8:
        raise ValueError("QSA uint8 K/V cache without block scales is not nvfp4")
    if _qfn_use_fp8 or _qfn_nvfp4:
        if k_scale is None or v_scale is None:
            raise ValueError("QSA quantized KV cache requires k_scale and v_scale")
        if k_scale.numel() != 1 or v_scale.numel() != 1:
            raise ValueError("QSA quantized KV cache expects scalar per-layer scales")
        if k_scale.device != q.device or v_scale.device != q.device:
            raise ValueError("QSA KV scales must live on the query device")
""",
    ),
    # --- Aufruf: Zeiger + Schrittweiten -------------------------------------
    (
        "wrapper-kachelprofil",
        """    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
""",
        """    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    # QFN-PATCH kvq-s2b -- HEBEL 3 in der URSPRUENGLICHEN Fassung (S2 2.5,
    # Ausweg 1): "BLOCK_N = 32 mit zwei Stufen statt 64 mit einer".
    #
    # Hebel 3a (num_stages=2 bei BLOCK_N=64) ist gemessen und verworfen
    # (0,938x-0,944x ueber vier Laengen, kvq-logs/s2b3a-kennlinie.log). Er
    # erledigt DIESE Fassung NICHT mit: bei halber Kachel halbiert sich auch
    # der Stufenpuffer, die Belegungsrechnung faellt also anders aus (S2 11.5,
    # zweiter Spiegelstrich).
    #
    # Diese Uebersteuerung ist AUSSCHLIESSLICH Messwerkzeug. Leer = die
    # Kachelwahl des Image, Wort fuer Wort unveraendert; die Vorgabe aendert
    # sich also NICHT. Sie greift NUR im breiten Kachelfall (block_n > 16) --
    # der Decode-Fall BLOCK_N=16 bleibt unangetastet, weil Hebel 3 die breite
    # Prefill-Kachel meint und sonst zwei Dinge zugleich variiert wuerden.
    #
    # Was dabei UNVERMEIDLICH mitwandert und deshalb hier steht statt
    # verschwiegen zu werden: num_tiles verdoppelt sich, damit auch
    # max_useful_splits -- num_splits kann folglich steigen, wo target_splits
    # bisher nicht die bindende Schranke war. Das ist keine zweite frei
    # gewaehlte Variable, sondern die Formel des Image; der Selbsttest weist
    # num_splits je Profil aus.
    _qfn_bn_env = _qfn_ops_os.environ.get("QFN_KVQ_WIDE_BLOCK_N", "").strip()
    if _qfn_bn_env and block_n > 16:
        block_n = int(_qfn_bn_env)
    # Zweites Messwerkzeug, gleiche Bauart und gleiche Bedingung: die Warpzahl
    # der BREITEN Kachel. Sie wird gebraucht, um Kachelbreite und Warpzahl zu
    # TRENNEN -- der Hebel-2-Fehler (S2 10.2) trifft die Profile mit
    # BLOCK_N=16 UND partial_warps=4 zugleich, und S2 10.3 nennt genau diese
    # Kombination als ungemessenen Verdaechtigen. Leer = Wahl des Image.
    _qfn_warps_env = _qfn_ops_os.environ.get("QFN_KVQ_WIDE_WARPS", "").strip()
    if _qfn_warps_env and partial_warps == 2:
        partial_warps = int(_qfn_warps_env)

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
""",
    ),
    (
        "wrapper-aufruf-zeiger",
        """    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
""",
        """    # QFN-PATCH kvq-s1: im bf16-Fall einen beliebigen gueltigen Zeiger
    # uebergeben (Triton braucht auch fuer den wegkompilierten Zweig ein
    # Argument); dasselbe Vorgehen wie in MiniMax-M3.
    _qfn_quant = _qfn_use_fp8 or _qfn_nvfp4
    _qfn_k_scale_arg = k_scale if _qfn_quant else q
    _qfn_v_scale_arg = v_scale if _qfn_quant else q
    _qfn_k_sf_arg = k_scale_cache if _qfn_nvfp4 else q
    _qfn_v_sf_arg = v_scale_cache if _qfn_nvfp4 else q
    _qfn_ks_strides = (
        (k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2))
        if _qfn_nvfp4
        else (0, 0, 0)
    )
    _qfn_vs_strides = (
        (v_scale_cache.stride(0), v_scale_cache.stride(1), v_scale_cache.stride(2))
        if _qfn_nvfp4
        else (0, 0, 0)
    )
    # QFN-PATCH kvq-s1/s2: Der quantisierte Zweig braucht mehr Shared Memory als
    # der bf16-Zweig -- die dequantisierten bf16-Kacheln stehen neben den
    # geladenen Rohkacheln. GEMESSEN auf SM120 (Serverstart 28.08. 08:47): mit
    # BLOCK_N=64 und num_stages=2 verlangte der fp8-Kernel 106.496 B, die Karte
    # stellt 101.376 B bereit -- 5.120 B zu wenig, und der Start stirbt beim
    # ersten echten Vorwaertslauf mit triton OutOfResources. Eine Pipeline-Stufe
    # weniger nimmt genau eine Kachelgeneration aus dem Puffer.
    #
    # Bewusst NUR fuer die breiten Kacheln: BLOCK_N=64 waehlt der Wrapper bei
    # >= 32 Programmen, also im Prefill/Chunked-Prefill. Der Decode-Fall faehrt
    # BLOCK_N=16, passt mit zwei Stufen bequem hinein und behaelt sie deshalb --
    # dort haengt die Latenz am Laden, und genau da soll die Pipeline bleiben.
    #
    # nvfp4 laedt zwar WENIGER Bytes als fp8 (128+16 statt 256 je Zeile), haelt
    # aber zusaetzlich float32-Zwischenwerte; die Grenze wird also nicht
    # automatisch weiter. Deshalb dieselbe Regel, und der Selbsttest faehrt alle
    # fuenf Kachel-Profile ab, statt sie zu erschliessen.
    # QFN-PATCH kvq-s2b: Uebersteuerung NUR fuer die Machbarkeitsprobe zu
    # Hebel 3a. Leer = die Regel aus Stufe 1, Wort fuer Wort unveraendert;
    # die Vorgabe aendert sich also NICHT. Gesetzt = fester Stufenwert,
    # damit messbar wird, ob num_stages=2 unter Hebel 1 ueberhaupt in den
    # Shared Memory passt (Stufe 1 kippte dort bei 106.496 gegen 101.376 B).
    #
    # QFN-PATCH kvq-s2b -- HEBEL 3a: GEPRUEFT UND VERWORFEN, die Regel bleibt.
    #
    # Die Vermutung war, `_qfn_quant` sei zu breit gefasst: gerissen hatte die
    # Grenze der fp8-Kernel (106.496 B gegen 101.376 B, gemessen 28.08.), und
    # fp8 laedt 256 Byte je Zeile, nvfp4 nur 144. Der Platz ist auch wirklich
    # da -- num_stages=2 fordert 75.776 B, erst drei Stufen sprengen die Grenze
    # (kvq-logs/s2b-stagesprobe.log).
    #
    # ABER: zwei Stufen sind fuer nvfp4 GEMESSEN LANGSAMER, und zwar konsistent
    # ueber vier Laengen -- 12,8k 0,938x, 65k 0,944x, 131k 0,944x, 196k 0,941x
    # (kvq-logs/s2b3a-kennlinie.log, s2b3a-arm-{vor,nach}.log). Die plausible
    # Ursache ist Belegung, nicht Kapazitaet: eine Stufe braucht ~10.240 B, also
    # passen rund neun Bloecke gleichzeitig auf einen SM; zwei Stufen brauchen
    # ~75.776 B und lassen nur noch einen zu. Der Pipelinegewinn zahlt den
    # Belegungsverlust nicht zurueck `[?]`.
    #
    # Die Regel aus Stufe 1 ist also fuer nvfp4 richtig -- aus einem ANDEREN
    # Grund als dem, aus dem sie geschrieben wurde. Sie bleibt unveraendert.
    # QFN_KVQ_STAGES ist ausschliesslich Messwerkzeug (leer = diese Regel).
    _qfn_stages_env = _qfn_ops_os.environ.get("QFN_KVQ_STAGES", "").strip()
    _qfn_stages = (
        int(_qfn_stages_env)
        if _qfn_stages_env
        else (1 if (_qfn_quant and block_n > 16) else 2)
    )
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        _qfn_k_scale_arg,
        _qfn_v_scale_arg,
        _qfn_k_sf_arg,
        _qfn_v_sf_arg,
        logical_indices,
""",
    ),
    (
        "wrapper-aufruf-strides",
        """        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
""",
        """        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        _qfn_ks_strides[0],  # QFN-PATCH kvq-s2
        _qfn_ks_strides[1],
        _qfn_ks_strides[2],
        _qfn_vs_strides[0],
        _qfn_vs_strides[1],
        _qfn_vs_strides[2],
        logical_indices.stride(0),
""",
    ),
    (
        "wrapper-aufruf-constexpr",
        """        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        num_stages=2,
    )
""",
        """        BLOCK_M=block_m,
        BLOCK_N=block_n,
        USE_FP8=_qfn_use_fp8,  # QFN-PATCH kvq-s1
        KV_QUANT_NVFP4=_qfn_nvfp4,  # QFN-PATCH kvq-s2
        V_SF_SWIZZLED=bool(v_sf_swizzled),  # QFN-PATCH kvq-s2
        SF_MODE=_QFN_SF_MODE,  # QFN-PATCH kvq-s2b (0/1/2 prod, 3/4/5 Diagnose)
        num_warps=partial_warps,
        num_stages=_qfn_stages,  # QFN-PATCH kvq-s1 (bf16: unveraendert 2)
    )
""",
    ),
    # --- S2b-Rueckfall: Software-Schreiber -----------------------------------
    (
        "software-schreiber",
        """def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
""",
        '''@triton.jit
def _qfn_nvfp4_store_kernel(
    key_ptr,
    value_ptr,
    k_data_ptr,
    k_sf_ptr,
    v_data_ptr,
    v_sf_ptr,
    slot_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_key_token,
    stride_key_head,
    stride_val_token,
    stride_val_head,
    s_kd_block,
    s_kd_token,
    s_kd_head,
    s_ks_block,
    s_ks_token,
    s_ks_head,
    s_vd_block,
    s_vd_token,
    s_vd_head,
    s_vs_block,
    s_vs_token,
    s_vs_head,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_SF_SWIZZLED: tl.constexpr,
) -> None:
    """QFN-PATCH kvq-s2: Software-Schreiber (Plan-Rueckfall S2b).

    Ein Programm je (Token, KV-Kopf, 16er-Gruppe) -- dieselbe Zerlegung wie
    `_reshape_cache_nvfp4_kernel` in PR #44389, damit hier nichts Neues
    erfunden wird. Quantisiert K und V nach NVFP4 und legt sie in DASSELBE
    Layout wie der vorkompilierte CUDA-Schreiber; der Leser oben muss deshalb
    nicht wissen, wer geschrieben hat.
    """
    tl.static_assert(HEAD_DIM % 16 == 0)
    SCALE_DIM: tl.constexpr = HEAD_DIM // 16
    tl.static_assert(SCALE_DIM % 4 == 0)
    tl.static_assert(PAGE_SIZE % 4 == 0)

    token = tl.program_id(0)
    head = tl.program_id(1)
    group = tl.program_id(2)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    if slot < 0:
        return
    block = slot // PAGE_SIZE
    offset = slot % PAGE_SIZE

    byte_offsets = tl.arange(0, 8)
    even = byte_offsets * 2
    odd = even + 1
    basis = group * 16
    k_even = tl.load(
        key_ptr + token * stride_key_token + head * stride_key_head + basis + even
    ).to(tl.float32)
    k_odd = tl.load(
        key_ptr + token * stride_key_token + head * stride_key_head + basis + odd
    ).to(tl.float32)
    v_even = tl.load(
        value_ptr + token * stride_val_token + head * stride_val_head + basis + even
    ).to(tl.float32)
    v_odd = tl.load(
        value_ptr + token * stride_val_token + head * stride_val_head + basis + odd
    ).to(tl.float32)

    # Die Layer-Skala wirkt als GLOBALE Skala: der Schreiber teilt durch sie,
    # der Leser multipliziert wieder damit. Startwert ist 1,0.
    k_global = tl.load(k_scale_ptr).to(tl.float32)
    v_global = tl.load(v_scale_ptr).to(tl.float32)
    k_quant = tl.where(k_global == 0.0, 1.0, 1.0 / k_global)
    v_quant = tl.where(v_global == 0.0, 1.0, 1.0 / v_global)

    k_max = tl.maximum(tl.max(tl.abs(k_even), axis=0), tl.max(tl.abs(k_odd), axis=0))
    v_max = tl.maximum(tl.max(tl.abs(v_even), axis=0), tl.max(tl.abs(v_odd), axis=0))
    k_sf_bits = _qfn_float_to_sf_bits((k_quant * k_max) / 6.0)
    v_sf_bits = _qfn_float_to_sf_bits((v_quant * v_max) / 6.0)
    k_sf_f = _qfn_sf_to_float(k_sf_bits)
    v_sf_f = _qfn_sf_to_float(v_sf_bits)
    k_out = tl.where(k_sf_f == 0.0, 0.0, k_quant / k_sf_f)
    v_out = tl.where(v_sf_f == 0.0, 0.0, v_quant / v_sf_f)

    k_low = _qfn_float_to_e2m1_bits(tl.clamp(k_even * k_out, -6.0, 6.0))
    k_high = _qfn_float_to_e2m1_bits(tl.clamp(k_odd * k_out, -6.0, 6.0))
    v_low = _qfn_float_to_e2m1_bits(tl.clamp(v_even * v_out, -6.0, 6.0))
    v_high = _qfn_float_to_e2m1_bits(tl.clamp(v_odd * v_out, -6.0, 6.0))

    tl.store(
        k_data_ptr
        + block * s_kd_block
        + offset * s_kd_token
        + head * s_kd_head
        + basis // 2
        + byte_offsets,
        k_low | (k_high << 4),
    )
    tl.store(
        v_data_ptr
        + block * s_vd_block
        + offset * s_vd_token
        + head * s_vd_head
        + basis // 2
        + byte_offsets,
        v_low | (v_high << 4),
    )

    ks_slot, ks_group = _qfn_sf_coord(offset, group, False, SCALE_DIM)
    tl.store(
        k_sf_ptr
        + block * s_ks_block
        + ks_slot * s_ks_token
        + head * s_ks_head
        + ks_group,
        k_sf_bits,
    )
    vs_slot, vs_group = _qfn_sf_coord(offset, group, V_SF_SWIZZLED, SCALE_DIM)
    tl.store(
        v_sf_ptr
        + block * s_vs_block
        + vs_slot * s_vs_token
        + head * s_vs_head
        + vs_group,
        v_sf_bits,
    )


def qsa_nvfp4_store(
    key: torch.Tensor,
    value: torch.Tensor,
    k_data: torch.Tensor,
    k_sf: torch.Tensor,
    v_data: torch.Tensor,
    v_sf: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    v_sf_swizzled: bool = True,
) -> None:
    """QFN-PATCH kvq-s2: NVFP4-Schreibpfad in Software (Rueckfall S2b).

    Wird nur benutzt, wenn `QFN_KVQ_NVFP4_WRITER=triton` gesetzt ist -- etwa
    weil der vorkompilierte CUDA-Schreiber im Image fehlt oder ein anderes
    Layout schreibt als der Leser erwartet. Beides misst kvq-nvfp4-test.py.
    """
    if not key.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA nvfp4 store requires CUDA and Triton")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("QSA nvfp4 store expects [tokens, heads, head_dim] K/V")
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError("QSA nvfp4 store expects BF16 inputs")
    if k_data.dtype != torch.uint8 or k_sf.dtype != torch.uint8:
        raise ValueError("QSA nvfp4 store expects uint8 cache pages")
    head_dim = key.shape[2]
    if head_dim % 16 or key.stride(2) != 1 or value.stride(2) != 1:
        raise ValueError("QSA nvfp4 store requires contiguous head_dim % 16 == 0")
    if k_data.shape[3] != head_dim // 2 or k_sf.shape[3] != head_dim // 16:
        raise ValueError("QSA nvfp4 store: cache page widths do not match head_dim")
    if k_data.stride(3) != 1 or k_sf.stride(3) != 1:
        raise ValueError("QSA nvfp4 store: cache rows must be contiguous")
    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return
    _qfn_nvfp4_store_kernel[(num_tokens, key.shape[1], head_dim // 16)](
        key,
        value,
        k_data,
        k_sf,
        v_data,
        v_sf,
        slot_mapping,
        k_scale,
        v_scale,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        k_data.stride(0),
        k_data.stride(1),
        k_data.stride(2),
        k_sf.stride(0),
        k_sf.stride(1),
        k_sf.stride(2),
        v_data.stride(0),
        v_data.stride(1),
        v_data.stride(2),
        v_sf.stride(0),
        v_sf.stride(1),
        v_sf.stride(2),
        PAGE_SIZE=k_data.shape[1],
        HEAD_DIM=head_dim,
        V_SF_SWIZZLED=bool(v_sf_swizzled),
        num_warps=1,
        num_stages=1,
    )


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
''',
    ),
]


# ---------------------------------------------------------------------------
# Datei 3 -- platforms/interface.py  (S2a, Nachtrag: die Seiten-Ausrichtung)
#
# WARUM DAS NOETIG IST -- GEMESSEN, NICHT VERMUTET
# ------------------------------------------------
# Erster nvfp4-Serverstart (28.08. 14:24 UTC) starb reproduzierbar mit
#
#   ValueError: CSA+linear mamba cache owner 'language_model.model.layers.0
#   .linear_attn' needs 1634304 bytes, but a main_kv tensor page has 921600
#   bytes.
#
# Beide Zahlen sind erklaerbar und belegen, dass der Patch selbst richtig war:
#   * 1.634.304 B ist die GDN-Zustandsseite (Conv 61.440 + SSM 1.572.864),
#     vorab aus config.json ausgerechnet -- sie stimmt aufs Byte.
#   * 921.600 B = 1600 x 576, also unsere KORREKTE nvfp4-Attention-Seite bei
#     Blockgroesse 1600. Die Spec-Geometrie (num_head_slots=4,
#     state_content_bytes=144) hat also gegriffen.
#
# Der Fehler liegt DAZWISCHEN: `Platform._align_hybrid_block_size` waehlt die
# Blockgroesse so, dass die Attention-Seite die GDN-Seite deckt -- und rechnet
# dafuer mit `backend_cls.customize_spec(...)`, wobei `backend_cls` aus
# `_find_non_ssm_backend()` kommt, also NICHT zwingend unser QSA-Backend ist.
# Ohne unsere customize_spec kam dort die ungepackte Rechnung heraus
# (2 Koepfe x (256+256) x 1 B = 1024 B/Token) und damit
#     16 x ceil(1.634.304 / (16 x 1024)) = 1600
# -- also exakt die fp8-Blockgroesse. Mit der richtigen Zahl 576 B/Token waeren
# es 16 x ceil(1.634.304 / 9.216) = 2848 gewesen, und 2848 x 576 = 1.640.448
# deckt die GDN-Seite.
#
# WARUM NICHT --block-size 2848 VON HAND: die Ausrichtung setzt anschliessend
# `mamba_page_size_padded = block_size x attn_page_size_1_token` -- mit ihrer
# falschen 1024 also 2.916.352 statt 1.640.448. Die spaetere Vereinheitlichung
# (kv_cache_utils.unify_kv_cache_spec_page_size) zieht dann ALLE Gruppen auf
# dieses Maximum hoch, und unsere Attention-Seite wuerde von 576 auf 1024
# B/Token gepolstert -- der gesamte Kapazitaetsgewinn waere weg, OHNE
# Fehlermeldung. Die Blockgroesse von Hand zu setzen behebt den Absturz und
# tarnt den Verlust. Deshalb wird die Rechnung selbst korrigiert.
#
# EINGRIFFSTIEFE: eine Zuweisung, hart gegated auf `kv_quant_mode.is_nvfp4`.
# Fuer jede andere Konfiguration ist der Zweig unerreichbar -- kein anderes
# Backend im Image akzeptiert nvfp4-KV ueberhaupt (FlashAttention fuehrt es gar
# nicht in supported_kv_cache_dtypes). Kein Modell ausser diesem kann den Pfad
# also betreten.
# ---------------------------------------------------------------------------

PLAT_EDITS = [
    (
        "hybrid-seitenausrichtung",
        """            attn_page_size_1_token = backend_cls.customize_spec(
                attn_spec
            ).page_size_bytes
""",
        """            _qfn_spec = backend_cls.customize_spec(attn_spec)
            # QFN-PATCH kvq-s2-plat: Wenn das gefundene Backend die
            # nvfp4-Packung nicht kennt (customize_spec liess
            # state_content_bytes auf None), hier nachziehen -- sonst richtet
            # sich die Blockgroesse nach einer Seite, die es gar nicht gibt.
            # Dieselbe Geometrie wie FlashInferBackend.customize_spec und wie
            # unser QSA-Backend: K und V als getrennte Head-Slots, Zeile
            # head//2 + head//16 Byte (fp4-Daten + fp8-Blockskalen).
            if _qfn_spec.state_content_bytes is None and kv_quant_mode.is_nvfp4:
                from dataclasses import replace as _qfn_replace
                from vllm.utils.torch_utils import (
                    get_dtype_size as _qfn_dtype_size,
                    nvfp4_kv_cache_full_dim as _qfn_nvfp4_full_dim,
                )

                _qfn_spec = _qfn_replace(
                    _qfn_spec,
                    num_head_slots=2 * _qfn_spec.num_kv_heads,
                    state_content_bytes=(
                        _qfn_nvfp4_full_dim(_qfn_spec.head_size)
                        * _qfn_dtype_size(_qfn_spec.dtype)
                    ),
                )
            attn_page_size_1_token = _qfn_spec.page_size_bytes
""",
    ),
]


def _patch_datei(
    pfad: str, edits: list[tuple[str, str, str]], tag: str, marker: str = MARKER
) -> int:
    if not os.path.exists(pfad):
        print(f"[kvq-patch] FEHLER: {pfad} fehlt", file=sys.stderr)
        return 1
    original = open(pfad, encoding="utf-8").read()
    if marker in original:
        print(f"[kvq-patch] bereits gepatcht: {pfad}")
        return 0

    # Erst ALLE Anker zaehlen, dann erst schreiben -- ein halb gepatchtes Image
    # waere schlimmer als ein ungepatchtes.
    for name, anker, _ in edits:
        n = original.count(anker)
        if n != 1:
            print(
                f"[kvq-patch] FEHLER ({tag}/{name}): Anker {n}x gefunden "
                "(erwartet 1x) -- Image weicht vom geprueften Stand ab "
                "(0.1.dev20073+g8e685d198), NICHT patchen.",
                file=sys.stderr,
            )
            return 1

    neu = original
    for _, anker, ersatz in edits:
        neu = neu.replace(anker, ersatz, 1)

    open(pfad, "w", encoding="utf-8").write(neu)
    try:
        py_compile.compile(pfad, doraise=True)
    except py_compile.PyCompileError as e:
        open(pfad, "w", encoding="utf-8").write(original)
        print(
            f"[kvq-patch] FEHLER beim Kompilieren ({tag}), zurueckgerollt: {e}",
            file=sys.stderr,
        )
        return 2

    print(f"[kvq-patch] {tag}: {len(edits)} Eingriffe angewandt auf {pfad}")
    for i, z in enumerate(neu.splitlines(), 1):
        if marker in z:
            print(f"[kvq-patch]   Marker Zeile {i}: {z.strip()[:88]}")
    return 0


def main() -> int:
    pfad_qsa = os.environ.get("QFN_KVQ_QSA_FILE", PFAD_QSA)
    pfad_ops = os.environ.get("QFN_KVQ_OPS_FILE", PFAD_OPS)
    pfad_plat = os.environ.get("QFN_KVQ_PLAT_FILE", PFAD_PLAT)

    # Zuerst die Plattform-Datei: sie hat genau einen Anker und ist die
    # wahrscheinlichste Drift-Stelle. Scheitert sie, ist noch nichts angefasst.
    rc = _patch_datei(pfad_plat, PLAT_EDITS, "platforms/interface.py", MARKER_PLAT)
    if rc:
        return rc
    rc = _patch_datei(pfad_qsa, QSA_EDITS, "qsa.py")
    if rc:
        return rc
    rc = _patch_datei(pfad_ops, OPS_EDITS, "ops/qsa.py")
    if rc:
        # qsa.py steht jetzt gepatcht da, ops/qsa.py nicht -- das waere ein
        # inkonsistentes Image. Zuruecknehmen ist hier nicht moeglich (die
        # Originale sind weg), aber der Server darf so nicht starten.
        print(
            "[kvq-patch] ABBRUCH: ops/qsa.py nicht patchbar, qsa.py aber schon. "
            "Container verwerfen und neu starten (Image auf der Platte ist "
            "unveraendert, der Patch lebt nur in der Container-Schicht).",
            file=sys.stderr,
        )
        return rc

    print(
        "[kvq-patch] Stufe 1 (fp8_e4m3) + Stufe 2 (nvfp4) angewandt, "
        "Skalen 1,0. Aktiv wird, was --kv-cache-dtype waehlt."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
