import torch

from dfd.av.heads import AlignmentHead, WindowHead, pair_features


def test_pair_features_stack_four_views():
    lips, audio = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    assert pair_features(lips, audio).shape == (2, 5, 32)


def test_window_head_gives_one_logit_per_window():
    head = WindowHead(width=8, hidden=16).eval()
    x = [torch.randn(3, 50, 8) for _ in range(3)]
    assert head(*x).shape == (3,)


def test_attention_lets_one_frame_move_the_window_score():
    torch.manual_seed(0)
    head = WindowHead(width=8, hidden=16).eval()
    x = [torch.zeros(1, 50, 8) for _ in range(3)]
    edited = [t.clone() for t in x]
    edited[0][0, 25] = 5.0
    with torch.no_grad():
        assert head(*x).item() != head(*edited).item()


def test_alignment_head_learns_which_frames_belong_together():
    # Audio is a noisy copy of the lips, so aligned pairs are learnable and rolled pairs are not.
    torch.manual_seed(0)
    head = AlignmentHead(width=8, hidden=32)
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for _ in range(200):
        lips = torch.randn(16, 20, 8)
        audio = lips + 0.1 * torch.randn_like(lips)
        positive, negative = head(lips, audio), head(lips, torch.roll(audio, 7, dims=1))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat([positive, negative]), torch.cat([torch.ones_like(positive), torch.zeros_like(negative)]))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    lips = torch.randn(4, 20, 8)
    audio = lips + 0.1 * torch.randn_like(lips)
    with torch.no_grad():
        aligned = head.window_score(lips, audio)
        shuffled = head.window_score(lips, torch.roll(audio, 7, dims=1))
    assert (shuffled > aligned).all()
