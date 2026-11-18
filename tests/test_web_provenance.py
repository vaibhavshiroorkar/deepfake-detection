"""The file check reads labels that AI tools write on purpose, and nothing else about the file."""

import json

from deepfake_detection.webapp import provenance

SEEDANCE_AIGC = json.dumps({"Label": "1", "ContentProducer": "00119144030008867405X210001", "ProduceID": "v0d8"})


def test_an_aigc_label_marked_1_is_found():
    found = provenance.labels_from_tags({"AIGC": SEEDANCE_AIGC, "encoder": "Lavf58.76.100"})
    assert len(found) == 1 and "AIGC" in found[0]


def test_an_aigc_label_marked_0_or_broken_is_not_evidence():
    assert provenance.labels_from_tags({"AIGC": json.dumps({"Label": "0"})}) == []
    assert provenance.labels_from_tags({"AIGC": "not json"}) == []


def test_codec_encoder_and_platform_tags_are_never_evidence():
    tags = {"encoder": "Lavf62.3.100", "handler_name": "ISO Media file produced by Google Inc.",
            "major_brand": "isom", "compatible_brands": "isomiso2avc1mp41"}
    assert provenance.labels_from_tags(tags) == []


def test_tag_names_are_read_without_regard_to_case():
    assert provenance.labels_from_tags({"aigc": SEEDANCE_AIGC})


def test_a_c2pa_manifest_declaring_ai_media_is_found():
    data = b"\x00\x00jumb....c2pa....http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia..."
    found = provenance.labels_from_bytes(data)
    assert len(found) == 1 and "C2PA" in found[0]


def test_a_c2pa_manifest_without_an_ai_source_is_not_evidence():
    assert provenance.labels_from_bytes(b"jumb c2pa digitalsourcetype/digitalCapture") == []
    assert provenance.labels_from_bytes(b"plain video bytes") == []
