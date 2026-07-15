#!/usr/bin/env bash
set -euo pipefail

cd /media/jgzn/SSD_lexar/RZ/UAVM/external/reloc3r_pairuav

python3 -m unittest \
  tests.test_phase27_a_v3_2a_identity_common \
  tests.test_phase27_a_v3_2a_identity_contract \
  tests.test_phase27_a_v3_2a_artifact_profile \
  tests.test_phase27_a_v3_2a_pairwise_join_matrix \
  tests.test_phase27_a_v3_2a_duplicate_missing_exports \
  tests.test_phase27_a_v3_2a_repair_candidates \
  tests.test_phase27_a_v3_2a_report -v

OUT=/tmp/phase27_a_v3_2a_identity_join_smoke
rm -rf "$OUT"
mkdir -p "$OUT"

cat > "$OUT/candidate.csv" <<'CSV'
canonical_pair_id,source_image_key,target_image_key,pair_key,pair_id,target_key,group_id,scene_key,source_row_index
c1,a.jpg,b.jpg,p1,pair1,t1,g1,s1,1
dup,a.jpg,c.jpg,p2,pair2,t1,g1,s1,2
dup,a2.jpg,c.jpg,p2,pair3,t1,g1,s1,3
,missing.jpg,z.jpg,p4,pair4,t2,g2,s2,4
dir,x.jpg,y.jpg,p5,pair5,t3,g3,s3,5
path,/root/IMG-001.JPG,/root/IMG-002.JPG,p6,pair6,t4,g4,s4,6
rowonly,,,,pair7,t5,g5,s5,7
CSV
cat > "$OUT/full.csv" <<'CSV'
canonical_pair_id,source_image_a,source_image_b,source_pair_key,source_group_id,source_json_id,source_row_index,baseline_surface_source
c1,a.jpg,b.jpg,p1,g1,j1,10,full
dup,a.jpg,c.jpg,p2,g1,j2,11,full
dup,a3.jpg,c.jpg,p2,g1,j3,12,full
dir,y.jpg,x.jpg,p5,g3,j4,13,full
path,img_001.png,img_002.png,p6,g4,j5,14,full
rowonly,,,,g5,j6,7,full
CSV
cat > "$OUT/stress.csv" <<'CSV'
canonical_pair_id,source_image_a,source_image_b,source_pair_key,source_group_id,source_json_id,source_row_index,baseline_surface_source
c1,a.jpg,b.jpg,p1,g1,j1,20,stress
stressonly,m.jpg,n.jpg,p9,g9,j9,21,stress
CSV

python3 -m scripts.phase27_a_v3_2a_identity_contract --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_artifact_profile --artifact candidate="$OUT/candidate.csv" --artifact full="$OUT/full.csv" --artifact stress="$OUT/stress.csv" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_pairwise_join_matrix --artifact candidate="$OUT/candidate.csv" --artifact full="$OUT/full.csv" --artifact stress="$OUT/stress.csv" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_duplicate_missing_exports --artifact candidate="$OUT/candidate.csv" --artifact full="$OUT/full.csv" --artifact stress="$OUT/stress.csv" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_repair_candidates --profile "$OUT/tables/artifact_identity_profile.csv" --pairwise "$OUT/tables/pairwise_join_matrix.csv" --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_report --output-dir "$OUT"
python3 -m scripts.phase27_a_v3_2a_write_record --output-dir "$OUT"

test -f "$OUT/contract/canonical_pair_identity_contract.md"
test -f "$OUT/reports/a_v3_2a_route_verdict.md"
test -f "$OUT/phase27_a_v3_2a_identity_join_audit_experiment_record.md"
echo A_V3_2A_IDENTITY_JOIN_SMOKE_OK
