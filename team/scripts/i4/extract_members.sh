#!/bin/bash
# E-009 feature extraction for the added speech encoders (commands of the E-009 entry). Resumable: finished jobs are skipped.
export PYTHONPATH=src
P=.venv/Scripts/python.exe
for C in AU-4 AU-6; do
  $P -m dfd.sweep.extract_audio --cand $C --variant trim_norm --out-root data/_features_trimnorm || exit 1
  $P -m dfd.sweep.extract_audio --cand $C --variant aug1 --out-root data/_features_aug --jobs asvspoof_train,asvspoof_dev || exit 1
  $P -m dfd.sweep.extract_audio --cand $C --variant aug2 --out-root data/_features_aug --jobs asvspoof_train || exit 1
  $P -m dfd.sweep.extract_audio --cand $C --variant aug3 --out-root data/_features_aug --jobs asvspoof_train || exit 1
done
echo EXTRACT_DONE
