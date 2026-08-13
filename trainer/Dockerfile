# trainer/Dockerfile — CPU-torch runner for the UNCHANGED nnue-trainer python trainers.
#
# S4's whole design point is that `python/mcts/train_selfplay.py` in
# ~/Project/nnue-trainer stays byte-identical: Rust replaces the Java *emitter*,
# not the trainer. So this image deliberately contains NO trainer code. The
# trainer checkout is bind-mounted at /trainer at run time (see
# trainer/roundtrip.sh), which also means editing it on the host takes effect
# immediately, with no rebuild.
#
# Slimmer than the reference ~/Project/nnue-trainer/Dockerfile.trainer on
# purpose: that one carries a full JDK + a primed ~/.m2 because it also runs the
# Java self-play/gauntlet loops. Here the Java side is Rust's job, so all we
# need is python + CPU torch.
#
# Debian base, not alpine: torch only publishes glibc (manylinux) wheels — on
# musl pip falls back to building torch from source.
FROM python:3.12-slim-bookworm

# Versions pinned to Dockerfile.trainer's so a net trained here is numerically
# the same net the server sidecar would produce. Override at build time if the
# upstream image ever moves:
#   docker build --build-arg TORCH_VERSION=2.13.0 --build-arg NUMPY_VERSION=2.5.1 ...
ARG TORCH_VERSION=2.13.0
ARG NUMPY_VERSION=2.5.1

# CPU-only wheels: the +cpu index keeps the ~2.5 GB of CUDA payload out.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
 && pip install --no-cache-dir "numpy==${NUMPY_VERSION}"

# train_selfplay.py does `import train_policy` — a sibling-module import that
# only resolves when python/mcts is on sys.path. Running the script by path adds
# its own directory, but PYTHONPATH makes `-m`, unittest discovery and any
# wrapper work too.
ENV PYTHONPATH=/trainer/python/mcts
ENV PYTHONDONTWRITEBYTECODE=1

# The trainer checkout mounts here (read-only); /work is the scratch volume that
# holds self-play rows, candidates and logs.
WORKDIR /trainer
VOLUME /work

# Deliberately no ENTRYPOINT: callers run explicit `python python/mcts/...`
# command lines, which is what trainer/README.md documents and what
# trainer/roundtrip.sh executes.
CMD ["python", "-c", "import torch; print('torch', torch.__version__, 'cpu-only')"]
