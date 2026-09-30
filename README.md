# RepresentationFM

Representation 연구를 위한 독립 프로젝트입니다. 실행 경계는 다음과 같습니다.

```text
현재 영상 / 실제 지시문 / proprioception
    → ObservationEncoder   [B, N, D] + padding mask
    → Representation       [B, Nz, D] + padding mask
    → Conditioning         [B, C]
    → FMActionHead         [B, action_horizon, action_dim]
```

`../DP`에서 conditional U-Net, straight-line FM, action normalization,
LIBERO 데이터/rollout 처리를 가져와 분리했습니다. 기존 DP 디렉토리에 런타임
의존성이 없으며, 기존 프로젝트의 파일·실험 결과를 변경하지 않습니다.
이는 새 입력 구조의 연구용 baseline이며 **아직 LIBERO-10 성공률이나 RTX 3090
학습시간을 측정한 모델이 아닙니다.** 기존 FM의 70.5%/약 2시간 수치를 이 모델의
결과로 사용할 수 없습니다.

## 모듈 경계

| 파일 | 역할 / 수정 지점 |
|---|---|
| `src/repr_fm/encoders.py` | Frozen DINOv2 spatial features, frozen BERT language tokens, trainable projections, camera/spatial/modality embeddings, proprio normalization |
| `src/repr_fm/interfaces.py` | `TokenBatch(tokens, padding_mask)` 공통 계약. Mask의 `True`는 padding |
| `src/repr_fm/representation.py` | 기본 multimodal Transformer와 `auxiliary_loss()` 연구 확장 지점 |
| `src/repr_fm/conditioning.py` | Representation 이후 masked mean pooling + MLP. Action head에 전달하는 유일한 관측 경로 |
| `src/repr_fm/action_head.py` | Condition/noisy actions/flow time만 받는 U-Net, FM loss, Euler sampling |
| `src/repr_fm/policy.py` | 네 모듈 연결, loss 결합, normalization, observation key 제한 |
| `src/repr_fm/data.py` | 현재 observation 및 동일 episode의 미래 action chunk, 실제 instruction 반환 |
| `src/repr_fm/checkpoint.py` | Trainable parameters, normalization buffers, EMA, schema 검증 |

기본 입력은 observation history 1, 카메라 2개, joint 7 + gripper 2의 proprio입니다.
Vision은 카메라별 DINOv2 patch grid를 각각 8×8로 평균 pooling합니다. 두 카메라를
합쳐 평균내지 않습니다. Native 16×16 patch 전체를 사용하려면
`encoder.vision_grid_size: 16`으로 바꾸고 feature cache도 새로 생성합니다.
Language는 task-ID embedding이 아닌 실제 문장을 BERT로 인코딩합니다.
문장별 frozen feature는 메모리에 cache하지만 새로운 문장도 인코딩할 수 있습니다.

기본 representation은 width 256, 2-layer, 4-head Transformer입니다. 모든 modality가
같은 token sequence의 attention에 참여합니다. 이후 readout이 2048차원 condition을
만들며, raw proprio나 task-ID가 representation을 우회하는 경로는 없습니다.
기존 U-Net 폭 `[512, 1024, 1536]`, prediction horizon 6, action dimension 7,
Euler sampling 4회는 유지합니다. 이 U-Net은 약 227M parameters이므로 작은 head로
오해하지 않아야 합니다. 폭 축소는 별도 성능 검증이 필요한 설정 변경입니다.

## 실행

이 워크스페이스에서는 frozen backbone 실행에 필요한 패키지가 있는
`turbovla-libero` 환경을 사용할 수 있습니다. 스크립트가 `src`를 추가하므로
editable 설치 없이 실행할 수 있습니다. 다른 환경은 `pyproject.toml`의 의존성과
LIBERO/MuJoCo 환경을 준비합니다. 기존 `diffusion_policy` 환경은 cached-feature
CPU 테스트에는 사용할 수 있지만, raw language 인코딩에는 `transformers`가 필요합니다.

```bash
cd /home/jack/practice/RepresentationFM
FM_PYTHON=/home/jack/miniforge3/envs/turbovla-libero/bin/python

# 기본 config는 로컬 DINOv2 코드/weight cache와 로컬 BERT cache를 사용합니다.
"$FM_PYTHON" scripts/train.py \
  --config configs/libero10_fm.yaml \
  --device cuda \
  --output-dir outputs/libero10_fm_baseline
```

기본 batch size 16은 시작 설정입니다. GPU 메모리/throughput을 측정해 조정해야 합니다.
`text_local_files_only: true`이므로 가중치가 없으면 다운로드 대신 오류를 냅니다.
다른 머신에서는 `vision_repository`, dataset 경로 및 text encoder 경로/revision을
맞춰야 합니다. Frozen pretrained weights는 checkpoint에 포함되지 않으므로 같은
encoder weights를 유지하고, 장기 실험에는 text revision과 로컬 vision 소스를 고정하세요.

반복 학습에서 frozen encoder 비용을 줄이려면 새 token cache를 생성합니다.

```bash
"$FM_PYTHON" scripts/cache_features.py \
  --config configs/libero10_fm.yaml \
  --output-dir .cache/libero10_tokens \
  --device cuda --batch-size 16

"$FM_PYTHON" scripts/train.py \
  --config configs/libero10_fm.yaml \
  --feature-cache-dir .cache/libero10_tokens \
  --output-dir outputs/libero10_fm_cached --device cuda
```

Cache에는 camera/spatial vision features와 text features만 저장합니다.
Trainable projection, representation, conditioning은 매 update 다시 계산합니다.
기존 DP의 평균 CLS cache는 사용할 수 없으며 잘못된 schema/encoder/data 조합은
오류로 처리합니다. 기본 float32 vision cache는 frame당 약 384 KiB,
10만 frame이면 약 36.6 GiB입니다. Cache 생성은 데이터 전체를 처리하므로
필요한 저장 공간과 생성 시간을 고려해 direct/cached 경로를 선택하세요.
Encoder fine-tuning이나 매번 바뀌는 image augmentation은 현재 frozen-cache
실험 범위에 포함되지 않습니다.

재개할 때는 같은 config, 총 학습 step, batch size를 사용합니다.

```bash
"$FM_PYTHON" scripts/train.py \
  --config configs/libero10_fm.yaml \
  --feature-cache-dir .cache/libero10_tokens \
  --output-dir outputs/libero10_fm_cached \
  --resume outputs/libero10_fm_cached/latest.pt --device cuda
```

Checkpoint에 optimizer/scheduler/scaler/EMA/RNG와 소비한 batch cursor를 저장합니다.
현재 deterministic preprocessing에서는 데이터 순서가 복원됩니다. 향후 worker별
무작위 augmentation을 추가하면 해당 RNG 복원도 추가해야 합니다.
기존 task-ID DP/FM checkpoint와 호환되지 않는 별도 schema입니다.

LIBERO-10 평가는 실제 task language와 공식 initial states를 사용합니다.

```bash
MUJOCO_GL=egl "$FM_PYTHON" scripts/evaluate.py \
  --checkpoint outputs/libero10_fm_cached/latest.pt \
  --libero-config-dir /home/jack/practice/CLaD/.cache/libero \
  --device cuda --num-envs 4 --rollouts-per-task 50 \
  --inference-steps 4 --execution-steps 6 \
  --output outputs/libero10_fm_cached/evaluation.json
```

`--inference-steps`는 ODE 적분 횟수, `--execution-steps`는 다시 관측하기 전에
실행하는 action 개수입니다. 둘은 독립적입니다. `--task-ids 0 --instruction '...'`로
하나의 task에서 다른 지시문을 입력할 수 있습니다. 이 옵션 자체가 새로운 task
성공 조건이나 OOD benchmark를 만드는 것은 아닙니다. 비교 실험에서는
num-envs, seed, initial states, execution horizon, sampling steps를 맞춥니다.

## Representation 기법 추가

`Representation.forward(TokenBatch) -> TokenBatch`를 유지하면서 내부 구조를
교체합니다. 출력 token 개수는 달라져도 되지만 width와 padding mask는 맞춰야 합니다.
새 클래스를 사용하는 경우 `build_policy()`와 config 복원 경로도 함께 등록해,
평가 시 같은 구조가 재구성되도록 합니다.

Auxiliary objective는 다음 hook에 구현합니다.

```python
def auxiliary_loss(self, encoded, represented, batch):
    # encoded/represented: TokenBatch
    # batch에는 학습 supervision도 접근 가능. forward에는 관측만 입력됨.
    return your_scalar_loss
```

전체 loss는 `flow_loss + representation_loss_weight * representation_loss`입니다.
기본 hook은 0이며 아직 특정 representation-learning 기법을 구현하지 않았습니다.
FM loss는 trainable encoder projection, representation, conditioning까지 역전파됩니다.
`Z.detach()` 또는 전체 conditioning cache를 사용하면 이 학습 경로가 끊깁니다.

공정한 실험에서는 같은 fusion 구조에서 auxiliary loss의 유무를 먼저 비교하세요.
`representation.kind: identity`는 fusion을 제거하는 구조 대조군이며,
loss만 끄는 capacity-matched 대조군과 의미가 다릅니다.

## 검증

```bash
cd /home/jack/practice/RepresentationFM
/home/jack/miniforge3/envs/diffusion_policy/bin/python -m pytest -q
```

테스트는 외부 모델 다운로드 없이 작은 synthetic feature를 사용합니다. Gradient,
padding, language/task-ID 분리, 관측에서 action target 배제, 한 번의 observation
encoding, sampling, normalization, strict raw/EMA checkpoint 복원을 확인합니다.
로컬의 기존 DP 소스가 있으면 동일 U-Net weights의 수치적 일치도 확인합니다.
이 검증은 학습 후 LIBERO 성공률이나 GPU 학습 benchmark를 대체하지 않습니다.

작성 시 확인한 결과: CPU 테스트 31개 통과, 실제 LIBERO 영상/지시문과 로컬
DINOv2/BERT를 사용한 소형 policy의 loss/backward/sampling 통과, raw 입력과
cached 입력의 encoder 출력 일치, 실제 2-frame token cache 생성/읽기 통과.
합성 데이터의 4-step 학습과 중간 checkpoint 재개 결과도 파라미터·EMA가
일치했습니다. 이 실행 환경에서는 CUDA를 사용할 수 없어 GPU 메모리/속도와
멀티워커 simulator rollout은 검증하지 않았습니다.

## 출처

U-Net/FM 및 LIBERO 실행 경로는 `../DP`의 CLaD-derived 구현에서 분리·수정했습니다.
Apache-2.0 LICENSE를 보존했습니다. Frozen DINOv2/BERT의 코드와 가중치는 각 원본
프로젝트의 라이선스를 따르며 이 저장소에 복사하지 않습니다.
