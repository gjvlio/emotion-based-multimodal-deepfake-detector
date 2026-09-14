import sys
import os
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Silence noisy output
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

def main():
    import torch
    import torch.nn.functional as F
    from webapp.model_service import ModelService

    print("Initializing ModelService...", flush=True)
    svc = ModelService()

    test_vids = [
        ("REAL_SPEECH", Path("webapp/uploads/trim_0_396_WIN_20260913_19_59_50_Pro.mp4")),
        ("REAL_SILENT", Path("webapp/uploads/trim_0_606_WIN_20260913_20_02_18_Pro.mp4")),
        ("FAKE_ROA", Path("webapp/uploads/trim_0_961_AQMFIb_9Giu-DlMR3gYzkfySV_SO27ezRXU6u7timYSEXztanB0xIuGux7hO6jrYkWuLzcLlgMs8NY9CKrZHop0LhDTbNMYCynREidcDgg.mp4")),
        ("FAKE_ELON", Path("webapp/uploads/trim_0_565_AQMt0xSu5_Fd_JN9T63C2Aca-2elRjpAi8Jcw1-LezQU9KVx5V-R7rklkc6Vmr8jDv2aElSsdUcVu8s6sBlBtnM4UI7-rMt8B59yEUUUhg.mp4")),
    ]

    TAU_0 = 0.425
    T = 0.45
    LOGIT_0 = math.log(TAU_0 / (1.0 - TAU_0))

    for ground_truth, v in test_vids:
        if not v.exists():
            continue
        print(f"\n{'='*60}")
        print(f"Testing [{ground_truth}]: {v.name}")
        print(f"{'='*60}", flush=True)
        clip_id = "eval_" + v.stem[:12]
        
        audio_val, inp_ids, att_m, kf_pix, transcript = svc._prepare_e2e_inputs(v, clip_id)
        has_speech = bool(transcript and len(transcript.strip()) > 0)
        
        with torch.no_grad():
            B, K, C, H, W = kf_pix.shape
            frames = kf_pix.view(B * K, C, H, W)
            vit_out = svc.model._vit(pixel_values=frames).last_hidden_state[:, 0, :]
            z_v_seq = vit_out.view(B, K, 768)
            
            w2v_out = svc.model._wav2vec(audio_val.float()).last_hidden_state
            w2v_emb = w2v_out.mean(dim=1)
            
            bert_out = svc.model._bert(input_ids=inp_ids, attention_mask=att_m)
            bert_emb = bert_out.last_hidden_state[:, 0, :]
            
            audio_text_seq = torch.stack([w2v_emb, bert_emb], dim=1)
            z_v_attn, _ = svc.model.cross_attn_v(query=z_v_seq, key=audio_text_seq, value=audio_text_seq)
            z_v_seq = svc.model.norm_v(z_v_seq + z_v_attn)
            at_attn, _ = svc.model.cross_attn_at(query=audio_text_seq, key=z_v_seq, value=z_v_seq)
            audio_text_seq = svc.model.norm_at(audio_text_seq + at_attn)
            
            w2v_emb_fused = audio_text_seq[:, 0, :]
            bert_emb_fused = audio_text_seq[:, 1, :]
            z_at_fused = torch.cat([w2v_emb_fused, bert_emb_fused], dim=-1)
            
            gru_out, _ = svc.model.vit_gru(z_v_seq)
            z_v = gru_out[:, -1, :]
            
            fused = svc.model.bilinear_fusion(z_at_fused, z_v)
            fused_proj = F.gelu(svc.model.proj_ln(svc.model.bilinear_proj(fused)))
            
            emo_b = svc.model.emotion_head_b(z_v)
            prob_b = F.softmax(emo_b, dim=-1)
            
            if not has_speech:
                prob_a = torch.zeros(1, 6, device=svc.device)
                prob_a[:, 0] = 1.0 # Neutral voice
                sarc = torch.zeros(1, 1, device=svc.device)
                # In natural speechlessness, there is no voice-face conflict
                delta = torch.zeros(1, 6, device=svc.device)
            else:
                emo_a = svc.model.emotion_head_a(z_at_fused)
                prob_a = F.softmax(emo_a, dim=-1)
                sarc = svc.model.sarcasm_head(z_at_fused)
                delta = torch.abs(prob_a - prob_b)
                
            outer = torch.bmm(prob_a.unsqueeze(2), prob_b.unsqueeze(1)).view(1, 36)
            combined = torch.cat([fused_proj, outer, delta, sarc], dim=-1)
            raw_logit = svc.model.classifier(combined).item()
            
            # Calibrated probability
            calibrated_logit = (raw_logit - LOGIT_0) / T
            calibrated_p_fake = torch.sigmoid(torch.tensor(calibrated_logit)).item()
            
            verdict = "FAKE" if calibrated_p_fake > 0.50 else "REAL"
            confidence = calibrated_p_fake * 100 if verdict == "FAKE" else (1.0 - calibrated_p_fake) * 100
            
            print(f"Transcript: '{transcript}'")
            print(f"Speech Detected: {has_speech}")
            print(f"Raw Logit: {raw_logit:.4f}")
            print(f"Calibrated P(fake): {calibrated_p_fake:.4f}")
            print(f"VERDICT: {verdict} (Confidence: {confidence:.1f}%) [Expected: {'REAL' if 'REAL' in ground_truth else 'FAKE'}]")

if __name__ == "__main__":
    main()
