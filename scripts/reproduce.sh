#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${SEARCH_R1_ROOT:-$PKG_DIR/.work}"
REPO="${SEARCH_R1_REPO:-$ROOT/repos/Search-R1}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEARCH_R1_COMMIT="${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}"

mkdir -p "$ROOT"/{models,data/retrieval,data/processed/nq,data/processed/hotpotqa,runs,logs,repos,wandb}

usage() {
  cat <<USAGE
Usage: bash scripts/reproduce.sh <command> [args]

Commands:
  setup
  prepare-data [1.5b|7b|14b|all]
  launch-retriever
  train <1.5b|7b|14b> <ppo|grpo_original|grpo_paper_format|grpo_soft_format>
  eval <1.5b|7b|14b> <tag> <checkpoint_path>
  all

Environment:
  SEARCH_R1_ROOT     Work directory, default: $PKG_DIR/.work
  CONDA_SH           Conda profile script, default: ~/miniconda3/etc/profile.d/conda.sh
  HF_ENDPOINT        Hugging Face endpoint/mirror, default: https://hf-mirror.com
  PIP_INDEX_URL      Pip index, default: https://pypi.org/simple
USAGE
}

model_id_for_scale() {
  local scale
  scale="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$scale" in
    1.5b|15b|1_5b) echo "Qwen/Qwen2.5-1.5B-Instruct" ;;
    7b) echo "Qwen/Qwen2.5-7B-Instruct" ;;
    14b) echo "Qwen/Qwen2.5-14B-Instruct" ;;
    *) echo "Unknown model scale: $1" >&2; exit 2 ;;
  esac
}

model_dir_for_scale() {
  local scale
  scale="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$scale" in
    1.5b|15b|1_5b) echo "$ROOT/models/Qwen2.5-1.5B-Instruct" ;;
    7b) echo "$ROOT/models/Qwen2.5-7B-Instruct" ;;
    14b) echo "$ROOT/models/Qwen2.5-14B-Instruct" ;;
    *) echo "Unknown model scale: $1" >&2; exit 2 ;;
  esac
}

scale_label() {
  local scale
  scale="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$scale" in
    1.5b|15b|1_5b) echo "1.5b" ;;
    7b) echo "7b" ;;
    14b) echo "14b" ;;
    *) echo "Unknown model scale: $1" >&2; exit 2 ;;
  esac
}

need_conda() {
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "Cannot find conda profile script: $CONDA_SH" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SH"
}

setup_repo() {
  if [[ ! -d "$REPO/.git" ]]; then
    git clone https://github.com/PeterGriffinJin/Search-R1.git "$REPO"
  fi
  git -C "$REPO" fetch --all --tags
  git -C "$REPO" checkout "$SEARCH_R1_COMMIT"
  bash "$PKG_DIR/scripts/apply_reward_patch.sh"
}

setup_envs() {
  need_conda
  if ! conda env list | awk '{print $1}' | grep -qx searchr1; then
    conda create -n searchr1 python=3.9 -y
  fi
  conda activate searchr1
  python -m pip install --upgrade pip setuptools wheel -i "$PIP_INDEX_URL"
  python -m pip install torch==2.4.0 vllm==0.6.3 transformers==4.47.1 datasets pandas pyarrow \
    accelerate sentencepiece protobuf requests fastapi uvicorn omegaconf hydra-core wandb \
    huggingface_hub modelscope -i "$PIP_INDEX_URL"
  python -m pip install -e "$REPO" -i "$PIP_INDEX_URL"

  if ! conda env list | awk '{print $1}' | grep -qx retriever; then
    conda create -n retriever python=3.10 -y
  fi
  conda activate retriever
  python -m pip install --upgrade pip setuptools wheel -i "$PIP_INDEX_URL"
  python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    transformers datasets pyserini uvicorn fastapi requests pandas pyarrow tqdm "numpy==1.26.4" \
    -i "$PIP_INDEX_URL"
  python -m pip install faiss-gpu-cu12==1.8.0.2 -i "$PIP_INDEX_URL" || \
    python -m pip install faiss-gpu==1.7.2 -i "$PIP_INDEX_URL"
}

download_model() {
  local scale="${1:-1.5b}"
  local model_id model_dir
  model_id="$(model_id_for_scale "$scale")"
  model_dir="$(model_dir_for_scale "$scale")"
  need_conda
  conda activate searchr1
  MODEL_ID="$model_id" MODEL_DIR="$model_dir" python - <<'PY'
import os
from modelscope import snapshot_download
from pathlib import Path
model_dir = Path(os.environ["MODEL_DIR"])
model_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(os.environ["MODEL_ID"], local_dir=str(model_dir))
print(model_dir)
PY
}

download_retrieval() {
  save="$ROOT/data/retrieval"
  mkdir -p "$save"
  download_file() {
    local url="$1"
    local out="$2"
    if [[ -s "$out" ]]; then
      echo "exists $out"
      return 0
    fi
    if command -v aria2c >/dev/null 2>&1; then
      aria2c --continue=true --max-connection-per-server=8 --split=8 \
        --dir "$save" --out "$(basename "$out")" "$url"
    else
      curl -L --fail --retry 20 --retry-all-errors -C - -o "$out" "$url"
    fi
  }
  download_file "$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_aa" "$save/part_aa"
  download_file "$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_ab" "$save/part_ab"
  download_file "$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz" "$save/wiki-18.jsonl.gz"
  [[ -f "$save/e5_Flat.index" ]] || cat "$save"/part_* > "$save/e5_Flat.index"
  [[ -f "$save/wiki-18.jsonl" ]] || gzip -dk "$save/wiki-18.jsonl.gz"
}

process_data() {
  need_conda
  conda activate searchr1
  export HF_ENDPOINT
  cd "$REPO"
  python scripts/data_process/nq_search.py --local_dir "$ROOT/data/processed/nq" --template_type base
  ROOT="$ROOT" python "$PKG_DIR/scripts/process_hotpotqa_search.py"
  ROOT="$ROOT" python "$PKG_DIR/scripts/merge_train_data.py"
}

prepare_data() {
  local scale="${1:-1.5b}"
  case "$(printf '%s' "$scale" | tr '[:upper:]' '[:lower:]')" in
    all)
      download_model 1.5b
      download_model 7b
      download_model 14b ;;
    *)
      download_model "$scale" ;;
  esac
  download_retrieval
  process_data
}

launch_retriever() {
  need_conda
  conda activate retriever
  cd "$REPO"
  export CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-0,1}"
  python search_r1/search/retrieval_server.py \
    --index_path "$ROOT/data/retrieval/e5_Flat.index" \
    --corpus_path "$ROOT/data/retrieval/wiki-18.jsonl" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu \
    2>&1 | tee "$ROOT/logs/retriever_server.log"
}

train_one() {
  local scale="${1:-}"
  local exp="${2:-}"
  [[ -n "$scale" && -n "$exp" ]] || { usage; exit 2; }
  scale="$(scale_label "$scale")"
  local model_dir
  model_dir="$(model_dir_for_scale "$scale")"
  need_conda
  conda activate searchr1
  cd "$REPO"
  export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
  export VLLM_ATTENTION_BACKEND=XFORMERS
  export PYTHONUNBUFFERED=1
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export WANDB_DIR="$ROOT/wandb"

  local adv=grpo reward=em mode="" lambda=0 n_agent=5 batch=24 minibatch=60 out="$ROOT/runs/${scale}_${exp}"
  case "$exp" in
    ppo)
      adv=gae; n_agent=1; batch=48; minibatch=48; reward=em ;;
    grpo_original)
      reward=em ;;
    grpo_paper_format)
      reward=custom; mode=paper_format; lambda=0.2 ;;
    grpo_soft_format)
      reward=custom; mode=soft_format; lambda=0.2 ;;
    *)
      echo "Unknown experiment: $exp" >&2
      exit 2 ;;
  esac
  batch="${TRAIN_BATCH_SIZE:-$batch}"
  minibatch="${PPO_MINI_BATCH_SIZE:-$minibatch}"
  local micro_batch="${PPO_MICRO_BATCH_SIZE:-6}"
  local tp_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
  local gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
  local max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
  local max_num_seqs="${VLLM_MAX_NUM_SEQS:-64}"

  if [[ "$reward" == custom ]]; then
    export SEARCH_R1_REWARD=custom
    export SEARCH_R1_REWARD_MODE="$mode"
    export SEARCH_R1_LAMBDA_F="$lambda"
  else
    unset SEARCH_R1_REWARD SEARCH_R1_REWARD_MODE SEARCH_R1_LAMBDA_F || true
  fi

  mkdir -p "$out" "$ROOT/logs" "$WANDB_DIR"
  python3 -m verl.trainer.main_ppo \
    data.train_files="$ROOT/data/processed/nq_hotpotqa_train.parquet" \
    data.val_files="$ROOT/data/processed/nq/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=256 \
    data.train_batch_size="$batch" \
    data.val_batch_size=48 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.adv_estimator="$adv" \
    actor_rollout_ref.model.path="$model_dir" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="$minibatch" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$micro_batch" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$micro_batch" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$tp_size" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$gpu_memory_utilization" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$max_num_batched_tokens" \
    actor_rollout_ref.rollout.max_num_seqs="$max_num_seqs" \
    actor_rollout_ref.rollout.n_agent="$n_agent" \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$micro_batch" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    "trainer.logger=[console,wandb]" \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=Search-R1 \
    trainer.experiment_name="${scale}_${exp}" \
    trainer.total_epochs=15 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-800}" \
    trainer.default_local_dir="$out" \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}" \
    retriever.topk=3 \
    2>&1 | tee "$ROOT/logs/${scale}_${exp}.log"
}

eval_one() {
  local scale="${1:-}"
  local tag="${2:-}"
  local ckpt="${3:-}"
  [[ -n "$scale" && -n "$tag" && -n "$ckpt" ]] || { usage; exit 2; }
  scale="$(scale_label "$scale")"
  need_conda
  conda activate searchr1
  cd "$REPO"
  unset SEARCH_R1_REWARD SEARCH_R1_REWARD_MODE SEARCH_R1_LAMBDA_F || true
  export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
  export VLLM_ATTENTION_BACKEND=XFORMERS
  local micro_batch="${PPO_MICRO_BATCH_SIZE:-6}"
  local tp_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
  local gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
  local max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
  local max_num_seqs="${VLLM_MAX_NUM_SEQS:-64}"
  mkdir -p "$ROOT/evals/logs"
  python3 -m verl.trainer.main_ppo \
    data.train_files="$ROOT/data/processed/nq/train.parquet" \
    data.val_files="$ROOT/data/processed/nq/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=48 \
    data.val_batch_size=48 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$ckpt" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$tp_size" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$gpu_memory_utilization" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$max_num_batched_tokens" \
    actor_rollout_ref.rollout.max_num_seqs="$max_num_seqs" \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$micro_batch" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$micro_batch" \
    trainer.logger=[] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.default_local_dir="$ROOT/evals/raw/$tag" \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}" \
    retriever.topk=3 \
    2>&1 | tee "$ROOT/evals/logs/${scale}_${tag}.log"
}

cmd="${1:-}"
shift || true
case "$cmd" in
  setup) setup_repo; setup_envs ;;
  prepare-data) prepare_data "${1:-1.5b}" ;;
  launch-retriever) launch_retriever ;;
  train) train_one "$@" ;;
  eval) eval_one "$@" ;;
  all)
    setup_repo
    setup_envs
    prepare_data all
    echo "Start retriever in another shell: bash scripts/reproduce.sh launch-retriever"
    for scale in 1.5b 7b 14b; do
      train_one "$scale" ppo
      train_one "$scale" grpo_original
      train_one "$scale" grpo_paper_format
      train_one "$scale" grpo_soft_format
    done ;;
  *) usage; exit 2 ;;
esac
