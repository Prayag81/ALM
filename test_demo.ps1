# =============================================================================
#  ALM — demo.py Test Commands
#  Run from the ALM root with venv activated:
#    .\venv\Scripts\activate
#    .\test_demo.ps1
# =============================================================================
#
#  File → Label reference:
#    alm_000001.wav  →  dog
#    alm_000005.wav  →  thunderstorm
#    alm_000018.wav  →  fireworks
#    alm_000025.wav  →  chainsaw
#    alm_000075.wav  →  engine
#    alm_000166.wav  →  siren
#    alm_002095.wav  →  street_music
#    alm_002107.wav  →  gun_shot
# =============================================================================

$AUDIO = "data/processed/audio"

# ── SINGLE-LABEL ──────────────────────────────────────────────────────────────

Write-Host "`n[1/9] Single-label: dog" -ForegroundColor Cyan
python demo.py --audio "$AUDIO/alm_000001.wav"

Write-Host "`n[2/9] Single-label: thunderstorm" -ForegroundColor Cyan
python demo.py --audio "$AUDIO/alm_000005.wav"

Write-Host "`n[3/9] Single-label: fireworks  (top-10)" -ForegroundColor Cyan
python demo.py --audio "$AUDIO/alm_000018.wav" --top-k 10

Write-Host "`n[4/9] Single-label: gun_shot" -ForegroundColor Cyan
python demo.py --audio "$AUDIO/alm_002107.wav"

# ── MULTI-LABEL: CLEAN ────────────────────────────────────────────────────────

Write-Host "`n[5/9] Multi-label clean: gun_shot" -ForegroundColor Magenta
python demo.py --audio "$AUDIO/alm_002107.wav" --mode multilabel

Write-Host "`n[6/9] Multi-label clean: dog" -ForegroundColor Magenta
python demo.py --audio "$AUDIO/alm_000001.wav" --mode multilabel

# ── MULTI-LABEL: MIXED ────────────────────────────────────────────────────────

Write-Host "`n[7/9] Multi-label mix: gun_shot + street_music  (alpha=0.5)" -ForegroundColor Yellow
python demo.py --audio "$AUDIO/alm_002107.wav" --mix "$AUDIO/alm_002095.wav" --mix-alpha 0.5 --mode multilabel

Write-Host "`n[8/9] Multi-label mix: dog + siren  (alpha=0.4)" -ForegroundColor Yellow
python demo.py --audio "$AUDIO/alm_000001.wav" --mix "$AUDIO/alm_000166.wav" --mix-alpha 0.4 --mode multilabel

# ── MULTI-LABEL: NOISY ────────────────────────────────────────────────────────

Write-Host "`n[9/9] Multi-label noisy: siren at SNR=10dB" -ForegroundColor Green
python demo.py --audio "$AUDIO/alm_000166.wav" --noise-snr 10 --mode multilabel

# ── BONUS: ALPHA SENSITIVITY (gun_shot + dog) ─────────────────────────────────

Write-Host "`n[BONUS] Alpha sensitivity: gun_shot + dog blended at 0.1 / 0.5 / 0.9" -ForegroundColor DarkCyan
python demo.py --audio "$AUDIO/alm_002107.wav" --mix "$AUDIO/alm_000001.wav" --mix-alpha 0.1 --mode multilabel
python demo.py --audio "$AUDIO/alm_002107.wav" --mix "$AUDIO/alm_000001.wav" --mix-alpha 0.5 --mode multilabel
python demo.py --audio "$AUDIO/alm_002107.wav" --mix "$AUDIO/alm_000001.wav" --mix-alpha 0.9 --mode multilabel

Write-Host "`nAll tests done." -ForegroundColor White
