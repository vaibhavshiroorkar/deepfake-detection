# Exhibition demo, 20 November

The live demo is the website's detector. You upload a short video of one person speaking, and in about 20 to 30 seconds it answers with one of four outcomes and shows which check raised it: the face, the voice or the lip sync. It runs on the home PC and you open it on the laptop through Tailscale.

## One-time setup (the 18th)

1. **Tailscale on both machines.** Install it from tailscale.com/download on the home PC and on the laptop, and log in on both with the same account. It is free for personal use. Nothing is opened to the public internet: only your own devices can reach the PC.
2. **Keep the home PC awake and unrestarted until the 20th.**
   - Never sleep, plugged in: `powercfg -change -standby-timeout-ac 0` (in a terminal on the PC).
   - Settings, Windows Update: pause updates for one week, so the PC does not restart overnight.
3. **Copy the demo clips to the laptop:** the 7 files of the running order below, from `data/demo/upload/` on the PC (upload these), and the same names from `data/demo/watch/` (play these to the examiners; FakeAVCeleb's own files do not play in a browser). They are research data: keep them on your own machines and do not share them.
4. **Test once from outside the house** (phone hotspot or college Wi-Fi): start the demo (below), open the address on the laptop, and check one clip.

## Start, check and stop (on the home PC)

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo_host.ps1 -Action start
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo_host.ps1 -Action status
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/demo_host.ps1 -Action stop

`start` loads the models (about a minute), waits until the page answers, and shares it. `status` prints the address to open on the laptop, of the form `https://<pc-name>.<tailnet>.ts.net/check`. Write it down; it does not change. Start it on the morning of the 20th before leaving, run `status`, and leave the PC on.

## Running order (about 5 minutes)

Each outcome below was measured on 11-17 with the same models and settings (`docs/demo-clips.csv`). These clips were **chosen to show each outcome**; they say nothing about accuracy, which the paper measures on thousands of clips.

| # | Clip (in `upload/`) | What it is | Expected answer | Raised by |
|---|---|---|---|---|
| 1 | `FakeAVCeleb_real_A__id07161__00159.mp4` | Real person, real voice | No signs found | |
| 2 | `FakeAVCeleb_cloned-voice_B__id00946__00126_fake.mp4` | Real face, cloned voice | Signs of manipulation | voice, lip sync |
| 3 | `FakeAVCeleb_FSGAN-face-swap_C__id00088__00005_id00060_gZ4RkcQ0SXE.mp4` | Face swapped (FSGAN), real voice | Signs of manipulation | face, lip sync |
| 4 | `FakeAVCeleb_lip-sync-Wav2Lip_C__id06443__00232_id09181_wavtolip.mp4` | Mouth regenerated to new speech (Wav2Lip) | Signs of manipulation | face, lip sync |
| 5 | `FakeAVCeleb_face-swap-+-Wav2Lip-cloned-voice_D__id00395__00420_id00823_KgExwCvgrUk_faceswap_id00897_wavtolip.mp4` | Face, mouth and voice all fake | Signs of manipulation | face, voice, lip sync |
| 6 | `HiFi-AVDF_real_hifiavdf__veo3.1__246.mp4` | Real clip from a modern corpus | No signs found | |
| 7 | `HiFi-AVDF_generated-sora2_hifiavdf__sora2__1305_generated.mp4` | Whole clip made by Sora 2 | No signs found (a miss) | |

**What to say at each step:**

1. "A real clip. The three checks place it among thousands of real clips; nothing stands out."
2. "The lips are real but the voice is cloned. The voice check flags it, and the lip-sync check sees that the lips do not match the words."
3. "A face swap with the real voice. The face check flags it. The voice is real, so the voice check does not."
4. "Wav2Lip redraws the mouth to fit new audio. The fit is close but not perfect, so the lip-sync check flags it."
5. "Everything is fake, and all three checks agree."
6. "A real clip from a newer dataset the models never saw: still no signs."
7. "Sora 2 makes the picture and the sound together, so the lips already match the voice. Our detector misses it. That is one of the paper's findings: on modern generators, accuracy drops close to chance. We report it instead of hiding it."

Then open "How sure?" on any result: it says how often real clips get this flag (about 5 in 100) and how few fakes from new sources it catches.

**If an examiner wants to try their own video:** record 5 to 10 seconds of one person talking to the camera on a phone, send it to the laptop, and upload it. Expect "no signs found" or "can't tell". The video is deleted right after the check.

## If something goes wrong

| Problem | What to do |
|---|---|
| The address does not open on the laptop | Check Tailscale is connected on the laptop. Try a phone hotspot (college Wi-Fi may block it). If the PC is unreachable, show the screenshots in `docs/demo/` and explain the run. |
| "The detector is busy" | Another check is running. Wait a minute and upload again. |
| A check takes over a minute | The PC may be doing something else. Wait; it answers within about two minutes. |
| An answer differs from the table | Small differences are possible after a restart (the face detector's results vary slightly). Say what the card shows and why. |
| The page shows an error | On the PC: `-Action stop`, then `-Action start`, wait a minute, try again. The log is in `runs/demo-host/server.err.log`. |

## The PPT

`deliverables/build/Deepfake_Detection_Presentation.pdf` and `.pptx`, in the college's order: title, introduction, need, literature review, three methodology slides, three results slides, discussion and future scope, conclusions, awards and publications, bibliography. Detailed slides follow as backup.
