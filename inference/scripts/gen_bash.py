configs = ["cfg", "no_cfg", "pag", "sag_0.3_cfg6.0", "sag_0.4_cfg5.0", "seg_sigma10_cfg3", "seg_sigmainf_cfg3", "slg"]
run = "05022026"
with open(f"run_{run}.sh", "w") as f:
    f.write("source /home/aaturevich/.venv/bin/activate\n")
    for config in configs:
        f.write(f"python /home/aaturevich/geneval/generation/diffusers_generate.py /home/aaturevich/geneval/prompts/prompts_05022026/evaluation_metadata.jsonl --config /home/aaturevich/geneval/model_configs/{config}.py --outdir /home/aaturevich/geneval/outputs/{config}_{run}\n")
        f.write(f"python /home/aaturevich/geneval/evaluation/evaluate_images.py /home/aaturevich/geneval/outputs/{config}_{run} --outfile /home/aaturevich/geneval/outputs/{config}_{run}/results.jsonl --model-path /home/aaturevich/geneval/models --model-config /home/aaturevich/geneval/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py\n")

