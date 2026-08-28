REPO="C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY="$REPO/.venv/Scripts/python.exe"
OUT="D:/MEE2024 output/MEE_output/cal_pileo_step2"
REFDIR="H:/Claude Code/HANDOFF_zenith_cubic/inpipeline_windowed"
# an array, not a string: REFDIR contains a space and word-splitting turns it into "H:\Claude"
REFS=()
for f in 08-11_Z1_base 08-11_Z2_mid_left 08-11_Z3_top_left 08-11_Z4_top_right 08-11_Z5_mid_right 08-11_Z6_bottom_right \
         08-12_Z1_base 08-12_Z2_mid_left 08-12_Z3_top_left 08-12_Z4_top_right 08-12_Z5_mid_right 08-12_Z6_bottom_right; do
  REFS+=("$REFDIR/$f.txt")
done

SITE=(--set observation_lat=42.740470 --set observation_long=-5.613780 --set observation_height=1101)
WX=(--set observation_temp=30.5 --set observation_pressure=896.6 --set observation_humidity=0.208 --set observation_wavelength=0.62)
CORR=(--set enable_corrections=True --set enable_corrections_ref=True --set enable_gravitational_def=False)

STAGE1_OPTS=(--set sensitive_mode_stack=True --set centroid_gaussian_subtract=True --set centroid_gaussian_thresh=4.0
--set min_area=2 --set sigma_subtract=0.0 --set delete_saturated_blob=False --set remove_edgy_centroids=True
--set centroid_refine_window=True --set centroid_window_sigma=2.0)

# stage2 <stage1_zip> <outdir> <tol> <obs_time>
stage2 () {
  local zip="$1" dir="$2" tol="$3" tim="$4"
  mkdir -p "$dir"
  "$PY" -m mee2024.cli distortion "$zip" --order cubic --date-from-header \
     --fix-distortion "${REFS[@]}" \
     --set distortion_fixed_coefficients=quadratic \
     --set distortion_fit_tol=$tol --set max_star_mag_dist=13 \
     "${CORR[@]}" --set observation_time=$tim "${SITE[@]}" "${WX[@]}" \
     --no-display --quiet -o "$dir" > "$dir/stage2.log" 2>&1
  local rc=$?
  local res=$(find "$dir" -name distortion_results.txt 2>/dev/null | head -1)
  if [ -n "$res" ]; then
    "$PY" -c "
import json,sys; d=json.load(open(sys.argv[1]))
print('  %-28s stars=%3d rms=%.4f ps=%.7f se=%6.2f ppm  t=%s' % (
  sys.argv[2], d['#stars used'], d['final rms error (arcseconds)'],
  d['platescale (arcseconds/pixel)'], d['platescale_relative_uncertainty']*1e6,
  d['observation_time (UTC)']))" "$res" "$(basename $dir)"
  else
    echo "  FAILED rc=$rc: $(basename $dir)"
  fi
}
