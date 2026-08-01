#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="/home/cunyuliu/rna_junction_preorganization_v1_1_20260801"
ARTIFACT_ROOT="/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801"
OUT_DIR="$ARTIFACT_ROOT/phase0/source_metadata"
RUN_ID="source_metadata_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$OUT_DIR/${RUN_ID}.log"

install -d -m 700 "$OUT_DIR"
umask 077
exec > >(tee "$RUN_LOG") 2>&1

echo "RUN_ID=$RUN_ID"
echo "TIME=$(date -Is)"
echo "HOST=$(hostname)"
echo "CODE_ROOT=$CODE_ROOT"
echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"
echo "PAYLOAD_POLICY=metadata_only"

get_public_json() {
  local name="$1"
  local url="$2"
  local data="$OUT_DIR/${name}.json"
  local headers="$OUT_DIR/${name}.headers"
  local provenance="$OUT_DIR/${name}.provenance"
  if [[ -e "$data" || -e "$headers" || -e "$provenance" ]]; then
    echo "TARGET_EXISTS=$name"
    return 3
  fi
  curl -L --fail --silent --show-error --max-time 60 \
    -D "$headers" -o "$data" "$url"
  local sha
  local size
  sha="$(sha256sum "$data" | awk '{print $1}')"
  size="$(stat -c %s "$data")"
  printf 'name=%s\nurl=%s\ndownloaded_at_utc=%s\nsize_bytes=%s\nsha256=%s\n' \
    "$name" "$url" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$size" "$sha" \
    > "$provenance"
  echo "DOWNLOADED name=$name size_bytes=$size sha256=$sha"
}

probe_public_endpoint() {
  local name="$1"
  local url="$2"
  local output="$OUT_DIR/${name}.probe"
  if [[ -e "$output" ]]; then
    echo "TARGET_EXISTS=$name"
    return 3
  fi
  local http_code
  local curl_rc=0
  http_code="$(curl -L --silent --show-error --max-time 30 -o /dev/null -w '%{http_code}' "$url")" || curl_rc=$?
  printf 'name=%s\nurl=%s\nprobed_at_utc=%s\nhttp_code=%s\ncurl_exit_code=%s\n' \
    "$name" "$url" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$http_code" "$curl_rc" \
    > "$output"
  echo "PROBED name=$name http_code=$http_code curl_exit_code=$curl_rc"
}

get_public_json \
  ncbi_bioproject_1188187 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=bioproject&id=1188187&retmode=json'

get_public_json \
  crossref_denny_2018 \
  'https://api.crossref.org/works/10.1016%2Fj.cell.2018.05.038'

get_public_json \
  crossref_dms_2026 \
  'https://api.crossref.org/works/10.1093%2Fnar%2Fgkag672'

probe_public_endpoint \
  figshare_api_27880434 \
  'https://api.figshare.com/v2/articles/27880434'

if [[ -e "$OUT_DIR/yesselman_dms_code_refs.txt" ]]; then
  echo "TARGET_EXISTS=yesselman_dms_code_refs"
  exit 3
fi
git ls-remote --heads \
  https://github.com/YesselmanLabPublications/2025_char_3d_struct_features.git \
  > "$OUT_DIR/yesselman_dms_code_refs.txt"
sha256sum "$OUT_DIR/yesselman_dms_code_refs.txt" \
  > "$OUT_DIR/yesselman_dms_code_refs.sha256"
echo "CODE_REFS_RECORDED=$OUT_DIR/yesselman_dms_code_refs.txt"

echo "STATUS=SOURCE_METADATA_COLLECTION_COMPLETE"
