"""Pinned upstream revisions. Every result is tied to these.

Change a pin only deliberately, and record why in docs/decisions.md.
"""

# Hugging Face dataset, revision read on 2026-09-18.
AURA_DATASET_REPO = "fzi-forschungszentrum-informatik/FZI-AURA"
AURA_DATASET_REVISION = "3404bd6b8fcd6eed53a0ec7610650a6393aabb49"

# SDK repo, HEAD on 2026-09-18.
AURA_SDK_REPO = "https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk"
AURA_SDK_COMMIT = "a36761db6c5ec4ad6e30064dce63a4437340a513"

# Model code repo, HEAD on 2026-09-18.
VGGT_OMEGA_REPO = "https://github.com/facebookresearch/vggt-omega"
VGGT_OMEGA_COMMIT = "a3ab0141f96838724423541044ff5ba301cfd36a"

# Gated Hugging Face model repo and checkpoint file names.
VGGT_OMEGA_HF_REPO = "facebook/VGGT-Omega"
VGGT_OMEGA_HF_REVISION = "1041e80fc0e911235d3426b0a3d9a81075111a53"  # read 2026-09-18
CHECKPOINT_PUBLIC_512 = "vggt_omega_1b_512.pt"
CHECKPOINT_RETRAINED_416 = "vggt_omega_1b_416_reproduce.pt"

# Comparison arm: the predecessor, VGGT. Code repo HEAD and weights revision read on 2026-09-18.
# facebook/VGGT-1B is CC BY-NC 4.0 and not gated.
VGGT_REPO = "https://github.com/facebookresearch/vggt"
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_HF_REPO = "facebook/VGGT-1B"
VGGT_HF_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
