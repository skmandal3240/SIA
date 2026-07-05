# Google Colab Free T4 — Step-by-Step

1. Open https://colab.research.google.com
2. Sign in with your Google account.
3. Click **File → Upload notebook**.
4. Upload `sia-lab/posttrain/SIA_P1_Training_Colab.ipynb` from the repo.
5. Click **Runtime → Change runtime type**.
6. Select **GPU** as Hardware accelerator, then click Save.
7. Run cells in order (play button on each cell, or `Ctrl+Enter`).
8. Wait ~2 hours. The final cell will download `sia_device_actions_lora.zip`.
9. Place the unzipped adapter into `sia-lab/posttrain/outputs/device_actions_lora/` in your local SIA repo.

## Common issues

- Session timeout: free Colab sessions expire after ~12 hours of idle. Keep the tab open.
- GPU not available: sometimes T4 is busy. Try again later or use Kaggle/RunPod.
- HuggingFace rate limit: set a free HF token in Colab secrets (`HF_TOKEN`) if downloads fail.

## Alternative providers

- **Kaggle**: free T4/P100, 30 GPU hours/week.
- **RunPod**: rent L4 at ~$0.25/hour. Run the same notebook or the direct command:
  ```bash
  python3 sia-lab/posttrain/sft.py --run --base unsloth/Llama-3.2-1B-Instruct
  ```
