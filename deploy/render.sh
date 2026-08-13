#!/usr/bin/env bash
# Render the deploy-branch docker-compose.yml from deploy/docker-compose.yml,
# pinning the image to an immutable reference.
#
# Usage:  deploy/render.sh <image-ref> [output-path]
#   e.g.  deploy/render.sh ghcr.io/korjavin/vsbot:sha-9602b2f… docker-compose.yml
#
# Called by .github/workflows/docker.yml for both the real deploy (master) and
# the dry run (pull requests), so the thing CI proves on a PR is the same code
# path that later rewrites the deploy branch. Runnable locally too — that is the
# point of having it be a script rather than an inline `sed` in the workflow.
#
# It is deliberately strict: it refuses to emit a compose file that still has a
# `build:` key, that does not contain the requested reference, or that had
# anything other than exactly one `image:` line to rewrite. A silently
# un-substituted template is the failure mode that ships `:latest` to
# production, so every one of those is a hard error, never a warning.
set -euo pipefail

usage() {
  echo "usage: $0 <image-ref> [output-path]" >&2
  exit 2
}

[ "$#" -ge 1 ] || usage
IMAGE_REF="$1"
OUT="${2:-docker-compose.yml}"
SRC="$(cd "$(dirname "$0")" && pwd)/docker-compose.yml"

case "$IMAGE_REF" in
  *:*) ;;
  *) echo "render: image reference '$IMAGE_REF' has no tag or digest" >&2; exit 1 ;;
esac

[ -f "$SRC" ] || { echo "render: template not found: $SRC" >&2; exit 1; }

image_lines=$(grep -c '^[[:space:]]*image:' "$SRC")
if [ "$image_lines" -ne 1 ]; then
  echo "render: expected exactly 1 'image:' line in $SRC, found $image_lines" >&2
  exit 1
fi

# `|` as the delimiter because the reference contains `/` and `:`. The
# replacement is applied to the single image line matched above, keeping its
# indentation.
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
sed -E "s|^([[:space:]]*image:[[:space:]]*).*$|\1${IMAGE_REF}|" "$SRC" >"$tmp"

grep -qF "$IMAGE_REF" "$tmp" || {
  echo "render: substitution produced no reference to $IMAGE_REF" >&2
  exit 1
}
if grep -qE '^[[:space:]]*build:' "$tmp"; then
  echo "render: rendered compose still has a 'build:' key — a remote host cannot build" >&2
  exit 1
fi
if grep -qE '^[[:space:]]*image:.*:latest[[:space:]]*$' "$tmp"; then
  echo "render: rendered compose still pins ':latest'" >&2
  exit 1
fi

mv "$tmp" "$OUT"
trap - EXIT
echo "render: wrote $OUT pinned to $IMAGE_REF"
