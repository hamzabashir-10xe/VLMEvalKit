# When using fp32 mlir

Make changes in following

1) smolvlm_vmfb.py -> 	New VMFB file path change, 
						e2e_runner updated file name
2) e2e_runner.py   -> pixel values change to npfloat32 datatype

# Running TOME merging algorithm 
VLMEVAL_SUBSET_N=350 VLMEVAL_SUBSET_SEED=42 python run.py --config smolvlm_tome.json


# Running normal benchmark
VLMEVAL_SUBSET_N=5 VLMEVAL_SUBSET_SEED=42 python run.py --data MMStar --model SmolVLM-500M-VMFB


