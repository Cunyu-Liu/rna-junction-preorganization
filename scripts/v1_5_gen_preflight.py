import json, hashlib, os, re, datetime, socket
RUN_ROOT=os.environ["RUN_ROOT"]; RUN_ID=os.environ["RUN_ID"]; BRANCH=os.environ["BRANCH"]; WORKTREE=os.environ["WORKTREE"]
PARENT_RUN="/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
PARENT_COMMIT="65896dca7181a172505b78416c171e6426094471"
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(65536),b''): h.update(b)
    return h.hexdigest()
utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
host=socket.gethostname()
env={
 "run_id":RUN_ID,"branch":BRANCH,"worktree":WORKTREE,"run_root":RUN_ROOT,
 "hostname":host,"utc":utc,"cst":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
 "os":"Linux 5.15.0-173-generic x86_64","python":"3.10.12 (system), rna_junction_preorganization_v1_1 conda env",
 "torch":"2.9.0+cu126","numpy":"2.2.6","scipy":"1.15.2","pandas":"2.3.3","pytest":"9.1.1","matplotlib":"3.10.9",
 "cuda_driver":"580.126.09","nvcc":"not found (pytorch cu126 build)","gpu_4_in_use":True,
 "gpu_awk_rule":"avoid GPU 4; B3 is CPU-sufficient per contract",
 "network":{
   "nature":True,"pmc":True,"oup":True,"plos":True,"biorxiv":True,
   "note":"oup returned 403 (bot-gate), biorxiv 429 (rate-limit); all reachable"
 },
 "disk":{"home_avail":"5.2T","mnt_avail":"12T"},
 "source_network_note":"authoritative sources reachable; PRIME is a 2026 preprint marked as such"
}
os.makedirs(f"{RUN_ROOT}/provenance",exist_ok=True)
with open(f"{RUN_ROOT}/provenance/preflight.json","w") as f:
    json.dump(env,f,indent=2,ensure_ascii=False)
# environment snapshot
with open(f"{RUN_ROOT}/provenance/environment_snapshot.json","w") as f:
    json.dump(env,f,indent=2,ensure_ascii=False)
# parent inventory (bounded) - key gate artifacts
gates={
 "contract":"contracts/1.4.docx",
 "T6":"sentinels/T6_TECTO_NEGATIVE_BOUND_AND_LOCKED.json",
 "Q6":"sentinels/Q6_QMAP_SOURCE_RECONSTRUCTED.json",
 "Q7":"sentinels/Q7_QMAP_TRANSFER_NOT_SUPPORTED.json",
 "N0":"sentinels/N0_METHODS_BOUNDARY_AUDIT.json",
 "B0":"sentinels/B0_BENCHMARK_FROZEN.json",
 "B1_fail":"sentinels/B1_FAILURE_MODE_VALIDATION_FAIL.json",
 "B1_pass":"sentinels/B1_FAILURE_MODE_VALIDATION_PASS.json",
 "B2":"sentinels/B2_POST_HOC_SENSITIVITY_COMPLETE.json",
 "R1":"sentinels/R1_RELEASE_SEALED.json",
 "M1":"sentinels/M1_MANUSCRIPT_DRAFT_AUTHORIZED.json",
 "E1":"sentinels/E1_E1_REPRODUCED_CLAIMS_ADMISSIBLE_SUBMISSION_READY.json",
 "state":"state/authoritative_status.json",
 "q7_metrics":"qmap/q7/metrics.json",
 "q7_decision":"qmap/q7/Q7_decision.json",
 "q6_decision":"qmap/q6/Q6_decision.json",
 "q6_membership":"qmap/q6/q6_membership.json",
 "n0_decision":"novelty/n0/N0_decision.json",
 "n0_paperspine":"novelty/n0/paper_spine.md",
 "n0_claim_matrix":"novelty/n0/claim_matrix.tsv",
}
rows=["gate\trelative_path\tsize\tmtime_utc\tsha256"]
for k,rel in gates.items():
    p=os.path.join(PARENT_RUN,rel)
    if os.path.exists(p):
        st=os.stat(p)
        rows.append(f"{k}\t{rel}\t{st.st_size}\t{datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t{sha(p)}")
    else:
        rows.append(f"{k}\t{rel}\tMISSING\t-\t-")
with open(f"{RUN_ROOT}/provenance/parent_inventory.tsv","w") as f:
    f.write("\n".join(rows)+"\n")
# sentinel inventory
srows=["sentinel\tstatus\tmtime_utc\tsha256"]
sd=f"{PARENT_RUN}/sentinels"
for fn in sorted(os.listdir(sd)):
    p=os.path.join(sd,fn); st=os.stat(p)
    status="FAIL" if re.search("FAIL",fn) else ("PASS" if re.search("PASS|COMPLETE|FROZEN|SEALED|AUDIT|LOCKED|AUTHORIZED|NOT_SUPPORTED|RECONSTRUCTED|READY",fn) else "other")
    srows.append(f"{fn}\t{status}\t{datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\t{sha(p)}")
with open(f"{RUN_ROOT}/provenance/parent_sentinel_inventory.tsv","w") as f:
    f.write("\n".join(srows)+"\n")
print("preflight files written")
