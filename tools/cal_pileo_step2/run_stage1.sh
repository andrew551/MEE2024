set -u
source "C:/Users/dpesm/AppData/Local/Temp/claude/C--Users-dpesm-OneDrive-Documents-GitHub-MEE2024/8dc8fd42-091a-4354-865c-ef39bab5567e/scratchpad/common.sh"
cd "$REPO"
R="I:/Leon 2026/2026-08-12/Eclipse/CAL_piLeo"

stage1 () {
  local dir="$OUT/$1"; shift
  mkdir -p "$dir"
  echo "=== stage1 $dir  ($# frames) ==="
  "$PY" -m mee2024.cli stack "$@" $STAGE1_OPTS --no-scan --no-exposure-check \
      --no-display --quiet -o "$dir" > "$dir/stage1.log" 2>&1
  echo "exit=$?  zip: $(ls "$dir"/centroid_data*.zip 2>/dev/null)"
}

# combined 17 (definitive, fresh)
stage1 s1_combined17 \
  "$R/18_29_19/CAL_piLeo_0000"{1,2,3,4,5,6}".fits" \
  "$R/18_29_27/CAL_piLeo_0000"{1,2,3,4,5,6,7,8}".fits" \
  "$R/18_29_51/CAL_piLeo_0000"{1,2,3}".fits"

# sub-stacks
stage1 s1_subA "$R/18_29_19/CAL_piLeo_0000"{1,2,3,4,5,6}".fits"
stage1 s1_subB "$R/18_29_27/CAL_piLeo_0000"{1,2,3,4,5,6,7,8}".fits"
stage1 s1_subC "$R/18_29_51/CAL_piLeo_0000"{1,2,3}".fits"

# 29 frames: adds the two 0.3 s pre-C3 blocks
stage1 s1_all29 \
  "$R/18_29_17/CAL_piLeo_0000"{1,2,3,4,5,6}".fits" \
  "$R/18_29_19/CAL_piLeo_0000"{1,2,3,4,5,6}".fits" \
  "$R/18_29_27/CAL_piLeo_0000"{1,2,3,4,5,6,7,8}".fits" \
  "$R/18_29_46/CAL_piLeo_0000"{1,2,3,4,5,6}".fits" \
  "$R/18_29_51/CAL_piLeo_0000"{1,2,3}".fits"
echo "ALL STAGE1 DONE"
