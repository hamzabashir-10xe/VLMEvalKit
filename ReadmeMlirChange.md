# When using fp32 mlir

Make changes in following

1) smolvlm_vmfb.py -> 	New VMFB file path change
2) e2e_runner.py   -> pixel values change to npfloat32 datatype

# When using Q8 vmfb file 
1) smolvlm_vmfb.py -> 	New VMFB file path change, 
2) Update gguf files path (since ne gguf files needed for Q8)

# Running TOME merging algorithm 
VLMEVAL_SUBSET_N=350 VLMEVAL_SUBSET_SEED=42 python run.py --config smolvlm_tome.json


# Running normal benchmark
VLMEVAL_SUBSET_N=5 VLMEVAL_SUBSET_SEED=42 python run.py --data MMStar --model SmolVLM-500M-VMFB

# Running one prompt with TOME ( current default r = 60 )
python test_tome.py
