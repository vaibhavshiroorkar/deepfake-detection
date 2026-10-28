$pp = 'D:\Final Year Project\deliverables_v2\presentation\Deepfake_Detection_Presentation_v2.pptx'
$out = 'D:\Final Year Project\submission\Multimodal_Deepfake_Detection_Presentation.pdf'
$a = New-Object -ComObject PowerPoint.Application
$p = $a.Presentations.Open($pp, -1, 0, 0)
$p.SaveAs($out, 32)
$p.Close(); $a.Quit()
