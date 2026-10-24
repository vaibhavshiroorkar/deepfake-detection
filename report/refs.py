"""Reference texts of the report (IEEE style). Every entry corresponds to a source that was opened and checked:
the papers of the literature matrix (status READ, or PARTIAL for the parts used in the text) and the encoder and
tool papers verified against the publisher record. Numbers in the text are assigned in order of first citation."""

REFS = {
    # face-video datasets and benchmarks
    "ffpp": "A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies and M. Nießner, “FaceForensics++: Learning to detect manipulated facial images,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.",
    "celebdf": "Y. Li, X. Yang, P. Sun, H. Qi and S. Lyu, “Celeb-DF: A large-scale challenging dataset for DeepFake forensics,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.",
    "celebdfpp": "Y. Li, D. Zhu, X. Cui and S. Lyu, “Celeb-DF++: A large-scale challenging video DeepFake benchmark for generalizable forensics,” arXiv:2507.18015v1, 2025.",
    "dfdc": "B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang and C. Canton Ferrer, “The DeepFake Detection Challenge (DFDC) dataset,” arXiv:2006.07397v4, 2020.",
    "bench": "Z. Yan, Y. Zhang, X. Yuan, S. Lyu and B. Wu, “DeepfakeBench: A comprehensive benchmark of deepfake detection,” in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.",
    # visual detectors
    "xray": "L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen and B. Guo, “Face X-ray for more general face forgery detection,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.",
    "sbi": "K. Shiohara and T. Yamasaki, “Detecting deepfakes with self-blended images,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.",
    "lips": "A. Haliassos, K. Vougioukas, S. Petridis and M. Pantic, “Lips don’t lie: A generalisable and robust approach to face forgery detection,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.",
    "identity": "S. Dong, J. Wang, R. Ji, J. Liang, H. Fan and Z. Ge, “Implicit identity leakage: The stumbling block to improving deepfake detection generalization,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.",
    "effort": "Z. Yan, J. Wang, P. Jin, K.-Y. Zhang, C. Liu, S. Chen, T. Yao, S. Ding, B. Wu and L. Yuan, “Orthogonal subspace decomposition for generalizable AI-generated image detection,” in Proc. 42nd International Conference on Machine Learning (ICML), PMLR 267, 2025.",
    "kornblith": "S. Kornblith, J. Shlens and Q. V. Le, “Do better ImageNet models transfer better?,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.",
    # generated images
    "cnnspot": "S.-Y. Wang, O. Wang, R. Zhang, A. Owens and A. A. Efros, “CNN-generated images are surprisingly easy to spot... for now,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.",
    "ojha": "U. Ojha, Y. Li and Y. J. Lee, “Towards universal fake image detectors that generalize across generative models,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.",
    "cozzolino": "D. Cozzolino, G. Poggi, R. Corvi, M. Nießner and L. Verdoliva, “Raising the bar of AI-generated image detection with CLIP,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024.",
    "fakejpeg": "P. Grommelt, L. Weiss, F.-J. Pfreundt and J. Keuper, “Fake or JPEG? Revealing common biases in generated image detection datasets,” in Computer Vision, ECCV 2024 Workshops, Lecture Notes in Computer Science, Springer, 2025.",
    # speech
    "asvspoof": "X. Wang, J. Yamagishi, M. Todisco, H. Delgado, A. Nautsch, N. Evans et al., “ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech,” Computer Speech and Language, 2020, doi: 10.1016/j.csl.2020.101114.",
    "aasist": "J. Jung, H. Heo, H. Tak, H. Shim, J. S. Chung, B. Lee, H. Yu and N. Evans, “AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.",
    "rawnet2": "H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans and A. Larcher, “End-to-end anti-spoofing with RawNet2,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021.",
    "w2vspoof": "H. Tak, M. Todisco, X. Wang, J. Jung, J. Yamagishi and N. Evans, “Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation,” in Proc. Speaker and Language Recognition Workshop (Odyssey), 2022.",
    "asv2021": "J. Yamagishi, X. Wang, M. Todisco, M. Sahidullah, J. Patino, A. Nautsch, X. Liu, K. A. Lee, T. Kinnunen, N. Evans and H. Delgado, “ASVspoof 2021: Accelerating progress in spoofed and deepfake speech detection,” in Proc. ASVspoof 2021 Workshop, 2021.",
    "inthewild": "N. M. Müller, P. Czempin, F. Dieckmann, A. Froghyar and K. Böttinger, “Does audio deepfake detection generalize?,” in Proc. Interspeech, 2022.",
    # audio-visual
    "fakeav": "H. Khalid, S. Tariq, M. Kim and S. S. Woo, “FakeAVCeleb: A novel audio-video multimodal deepfake dataset,” in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2021.",
    "dissonance": "K. Chugh, P. Gupta, A. Dhall and R. Subramanian, “Not made for each other: Audio-visual dissonance-based deepfake detection and localization,” in Proc. 28th ACM International Conference on Multimedia, 2020.",
    "zhou": "Y. Zhou and S.-N. Lim, “Joint audio-visual deepfake detection,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2021, doi: 10.1109/ICCV48922.2021.01453.",
    "lavdf": "Z. Cai, K. Stefanov, A. Dhall and M. Hayat, “Do you really mean that? Content driven audio-visual deepfake dataset and multimodal method for temporal forgery localization,” in Proc. International Conference on Digital Image Computing: Techniques and Applications (DICTA), 2022.",
    "avdf1m": "Z. Cai, S. Ghosh, A. P. Adatia, M. Hayat, A. Dhall, T. Gedeon and K. Stefanov, “AV-Deepfake1M: A large-scale LLM-driven audio-visual deepfake dataset,” in Proc. 32nd ACM International Conference on Multimedia, 2024, doi: 10.1145/3664647.3680795.",
    # physiological signals
    "fakecatcher": "U. A. Ciftci, I. Demir and L. Yin, “FakeCatcher: Detection of synthetic portrait videos using biological signals,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020, doi: 10.1109/TPAMI.2020.3009287.",
    "hearts": "U. A. Ciftci, I. Demir and L. Yin, “How do the hearts of deep fakes beat? Deep fake source detection via interpreting residuals with biological signals,” in Proc. IEEE International Joint Conference on Biometrics (IJCB), 2020.",
    "onphys": "J. Hernandez-Ortega, R. Tolosana, J. Fierrez and A. Morales, “DeepFakesON-Phys: DeepFakes detection based on heart rate estimation,” arXiv:2010.00400v3, 2020.",
    "heart": "C. Seibold, E. L. Wisotzky, A. Beckmann, B. Kossack, A. Hilsmann and P. Eisert, “High-quality deepfakes have a heart!,” Frontiers in Imaging, vol. 4, 2025, doi: 10.3389/fimag.2025.1504551.",
    "harraq": "O. Harraq and T. Aldwairi, “Physiological signals as a forensic modality for talking-face deepfake detection,” arXiv:2607.21776v1, 2026 (preprint).",
    "harraq2": "O. Harraq and T. Aldwairi, “Complementary rPPG-derived and lip-region frequency cues for talking-face deepfake detection,” arXiv:2609.22284v1, 2026 (preprint).",
    "lin": "Y. Lin, E. Maiorana, B. Li and P. Campisi, “Detection of photoplethysmography manipulation in video forgery,” in Proc. European Signal Processing Conference (EUSIPCO), 2023.",
    "deeprhythm": "H. Qi, Q. Guo, F. Juefei-Xu, X. Xie, L. Ma, W. Feng, Y. Liu and J. Zhao, “DeepRhythm: Exposing DeepFakes with attentional visual heartbeat rhythms,” in Proc. 28th ACM International Conference on Multimedia, 2020.",
    "pos": "W. Wang, A. C. den Brinker, S. Stuijk and G. de Haan, “Algorithmic principles of remote PPG,” IEEE Transactions on Biomedical Engineering, vol. 64, no. 7, pp. 1479-1491, 2017.",
    "mcduff": "D. J. McDuff, E. B. Blackford and J. R. Estepp, “The impact of video compression on remote cardiac pulse measurement using imaging photoplethysmography,” (venue not printed in the source read).",
    "cohface": "C. Yang, Y. Ouzar and L. Boubchir, “Beyond ambiguous visual cues: Studying physiological disruptions and cross-modal inconsistencies in deepfake videos,” arXiv:2609.12668v1, 2026 (preprint).",
    # calibration, adaptation, encoders and tools
    "calib": "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. 34th International Conference on Machine Learning (ICML), PMLR 70, 2017.",
    "lora": "E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. International Conference on Learning Representations (ICLR), 2022.",
    "dinov2": "M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.",
    "clip": "A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry et al., “Learning transferable visual models from natural language supervision,” in Proc. 38th International Conference on Machine Learning (ICML), PMLR 139, 2021.",
    "siglip": "X. Zhai, B. Mustafa, A. Kolesnikov and L. Beyer, “Sigmoid loss for language image pre-training,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2023.",
    "effnet": "M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. 36th International Conference on Machine Learning (ICML), PMLR 97, 2019.",
    "xception": "F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.",
    "wav2vec2": "A. Baevski, Y. Zhou, A. Mohamed and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020.",
    "wavlm": "S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen et al., “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505-1518, 2022.",
    "hubert": "W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov and A. Mohamed, “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451-3460, 2021.",
    "whisper": "A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. 40th International Conference on Machine Learning (ICML), PMLR 202, 2023.",
    "yunet": "W. Wu, H. Peng and S. Yu, “YuNet: A tiny millisecond-level face detector,” Machine Intelligence Research, vol. 20, no. 5, pp. 656-665, 2023, doi: 10.1007/s11633-023-1423-y.",
}
# the McDuff entry: venue not retrieved, so it is cited by title only and is not used in the text
REFS.pop("mcduff")
